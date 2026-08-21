// depth_hitl — hardware-in-the-loop depth renderer for the ground station.
//
// Listens for the OptiTrack pose datagram, ray-traces the baked obstacle scene
// from that pose on the CPU, and publishes the CNN-ready depth tensor over UDP
// at a fixed rate (10 Hz by default), with an optional terminal preview.
//
// The render pipeline is a port of the JAX one the policy trained against —
// see render.cpp. Verify parity with hitl/tools/compare_with_jax.py.

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <memory>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sched.h>
#include <sys/mman.h>
#include <termios.h>
#include <time.h>
#include <unistd.h>

#include "mocap.hpp"
#include "netout.hpp"
#include "preview.hpp"
#include "render.hpp"
#include "scene.hpp"

namespace {

volatile std::sig_atomic_t g_stop = 0;
void on_signal(int) { g_stop = 1; }

struct Options {
    std::string scene_path;
    std::string scene_dir;

    std::string in_addr   = "0.0.0.0";
    uint16_t    in_port   = 5005;      // relay.cpp's OPTITRACK_PORT
    int         rb_id     = -1;        // -1 = accept any rigid body

    std::string out_host  = "127.0.0.1";
    uint16_t    out_port  = 5010;
    bool        raw_out   = false;
    uint16_t    raw_port  = 5011;

    double      rate_hz   = 0.0;       // 0 = take cam_hz from the scene file
    bool        realtime  = false;
    bool        preview   = true;
    bool        print_pooled = false;
    int         pose_timeout_ms = 200;

    CamMount    mount;
    float       mount_rpy[3] = {0.0f, 0.0f, 0.0f};

    // Offline / debug modes.
    std::string one_pose;
    std::string poses_file;
    std::string dump_path;
    int         sniff     = 0;

    // Camera overrides (applied on top of the scene file's camera block).
    double fov = -1, min_range = -1, max_range = -1, quant = -1;
    int    width = -1, height = -1, pool = -1;
};

void usage() {
    std::printf(
"depth_hitl — HITL depth renderer (mocap pose → CNN depth tensor over UDP)\n"
"\n"
"Usage: depth_hitl --scene FILE [options]\n"
"\n"
"Scene\n"
"  --scene FILE         baked scene JSON (hitl/tools/export_scene.py)\n"
"  --scene-dir DIR      directory the 'n' hotkey cycles through\n"
"\n"
"Pose input (OptiTrack datagram, see mocap.hpp)\n"
"  --in-port N          UDP port to listen on            [5005]\n"
"  --in-addr ADDR       interface to bind                [0.0.0.0]\n"
"  --rb-id N            only accept this streaming id    [any]\n"
"  --pose-timeout-ms N  flag the frame stale after this  [200]\n"
"\n"
"Depth output\n"
"  --out-host HOST      destination for the depth tensor [127.0.0.1]\n"
"  --out-port N                                          [5010]\n"
"  --raw-out            also publish the full-res uint16 mm frame\n"
"  --raw-port N                                          [5011]\n"
"\n"
"Timing\n"
"  --rate HZ            render rate    [the scene file's camera cam_hz]\n"
"  --rt                 SCHED_FIFO + mlockall (needs privileges)\n"
"\n"
"Camera mount (rigid-body frame → camera; defaults match the simulator)\n"
"  --cam-offset x,y,z   camera offset in body FRD metres [0,0,0]\n"
"  --cam-rpy r,p,y      camera rotation in degrees       [0,0,0]\n"
"\n"
"Camera overrides (the scene file normally supplies these)\n"
"  --fov DEG  --width N  --height N  --min-range M  --max-range M\n"
"  --quant M  --pool N\n"
"\n"
"Display\n"
"  --no-preview         disable the terminal preview\n"
"  --print-pooled       print the pooled tensor under the preview\n"
"\n"
"Offline / debug\n"
"  --sniff N            hexdump N incoming pose packets and exit\n"
"  --pose \"x y z qw qx qy qz\"   render one pose and exit (no networking)\n"
"  --poses FILE         render one pose per line and exit\n"
"  --dump FILE          with --pose/--poses: write raw+pooled float32 frames\n"
"\n"
"Hotkeys while running: q quit  r reload scene  n next scene  p preview  s save PGM\n");
}

[[noreturn]] void die(const std::string& msg) {
    std::fprintf(stderr, "depth_hitl: %s\n", msg.c_str());
    std::exit(1);
}

const char* need(int argc, char** argv, int& i, const char* flag) {
    if (i + 1 >= argc) die(std::string("missing value for ") + flag);
    return argv[++i];
}

void parse_triplet(const char* s, float out[3], const char* flag) {
    if (std::sscanf(s, "%f,%f,%f", &out[0], &out[1], &out[2]) != 3)
        die(std::string(flag) + " expects x,y,z");
}

Options parse_args(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--scene")        o.scene_path = need(argc, argv, i, "--scene");
        else if (a == "--scene-dir")    o.scene_dir  = need(argc, argv, i, "--scene-dir");
        else if (a == "--in-port")      o.in_port    = static_cast<uint16_t>(std::atoi(need(argc, argv, i, "--in-port")));
        else if (a == "--in-addr")      o.in_addr    = need(argc, argv, i, "--in-addr");
        else if (a == "--rb-id")        o.rb_id      = std::atoi(need(argc, argv, i, "--rb-id"));
        else if (a == "--pose-timeout-ms") o.pose_timeout_ms = std::atoi(need(argc, argv, i, "--pose-timeout-ms"));
        else if (a == "--out-host")     o.out_host   = need(argc, argv, i, "--out-host");
        else if (a == "--out-port")     o.out_port   = static_cast<uint16_t>(std::atoi(need(argc, argv, i, "--out-port")));
        else if (a == "--raw-out")      o.raw_out    = true;
        else if (a == "--raw-port")     o.raw_port   = static_cast<uint16_t>(std::atoi(need(argc, argv, i, "--raw-port")));
        else if (a == "--rate")         o.rate_hz    = std::atof(need(argc, argv, i, "--rate"));
        else if (a == "--rt")           o.realtime   = true;
        else if (a == "--no-preview")   o.preview    = false;
        else if (a == "--print-pooled") o.print_pooled = true;
        else if (a == "--cam-offset")   parse_triplet(need(argc, argv, i, "--cam-offset"), o.mount.offset_body, "--cam-offset");
        else if (a == "--cam-rpy")      parse_triplet(need(argc, argv, i, "--cam-rpy"), o.mount_rpy, "--cam-rpy");
        else if (a == "--fov")          o.fov        = std::atof(need(argc, argv, i, "--fov"));
        else if (a == "--width")        o.width      = std::atoi(need(argc, argv, i, "--width"));
        else if (a == "--height")       o.height     = std::atoi(need(argc, argv, i, "--height"));
        else if (a == "--min-range")    o.min_range  = std::atof(need(argc, argv, i, "--min-range"));
        else if (a == "--max-range")    o.max_range  = std::atof(need(argc, argv, i, "--max-range"));
        else if (a == "--quant")        o.quant      = std::atof(need(argc, argv, i, "--quant"));
        else if (a == "--pool")         o.pool       = std::atoi(need(argc, argv, i, "--pool"));
        else if (a == "--sniff")        o.sniff      = std::atoi(need(argc, argv, i, "--sniff"));
        else if (a == "--pose")         o.one_pose   = need(argc, argv, i, "--pose");
        else if (a == "--poses")        o.poses_file = need(argc, argv, i, "--poses");
        else if (a == "--dump")         o.dump_path  = need(argc, argv, i, "--dump");
        else if (a == "-h" || a == "--help") { usage(); std::exit(0); }
        else die("unknown option '" + a + "' (try --help)");
    }
    return o;
}

void apply_overrides(const Options& o, CamCfg& cam) {
    if (o.fov       > 0) cam.fov_deg        = static_cast<float>(o.fov);
    if (o.width     > 0) cam.width          = o.width;
    if (o.height    > 0) cam.height         = o.height;
    if (o.min_range >= 0) cam.min_range     = static_cast<float>(o.min_range);
    if (o.max_range > 0) cam.max_range      = static_cast<float>(o.max_range);
    if (o.quant     >= 0) cam.quantization_m = static_cast<float>(o.quant);
    if (o.pool      > 0) cam.pool           = o.pool;

    if (cam.width <= 0 || cam.height <= 0 || cam.pool <= 0)
        die("camera width/height/pool must be positive");
    if (cam.width % cam.pool || cam.height % cam.pool)
        die("camera width and height must be divisible by pool (VALID max-pool, as in training)");
    if (!(cam.max_range > cam.min_range))
        die("max_range must exceed min_range");
    if (!(cam.cam_hz > 0.0f))
        die("camera cam_hz must be positive");
}

// ---------------------------------------------------------------------------
// Terminal raw mode for the hotkeys
// ---------------------------------------------------------------------------

struct RawTerminal {
    termios saved{};
    bool    active = false;

    void begin() {
        if (!isatty(STDIN_FILENO)) return;
        if (tcgetattr(STDIN_FILENO, &saved) != 0) return;
        termios raw = saved;
        raw.c_lflag &= ~(ICANON | ECHO);
        raw.c_cc[VMIN]  = 0;
        raw.c_cc[VTIME] = 0;
        if (tcsetattr(STDIN_FILENO, TCSANOW, &raw) != 0) return;
        int flags = fcntl(STDIN_FILENO, F_GETFL, 0);
        fcntl(STDIN_FILENO, F_SETFL, flags | O_NONBLOCK);
        active = true;
    }

    void end() {
        if (!active) return;
        tcsetattr(STDIN_FILENO, TCSANOW, &saved);
        active = false;
    }

    int poll_key() {
        if (!active) return -1;
        char c;
        ssize_t n = read(STDIN_FILENO, &c, 1);
        return n == 1 ? static_cast<unsigned char>(c) : -1;
    }
};

void quat_to_rpy_deg(const float q[4], float rpy[3]) {
    const float w = q[0], x = q[1], y = q[2], z = q[3];
    rpy[0] = std::atan2(2.0f * (w*x + y*z), 1.0f - 2.0f * (x*x + y*y));
    rpy[1] = std::atan2(2.0f * (w*y - z*x),
                        std::sqrt(std::max(0.0f, 1.0f - 4.0f * (w*y - z*x) * (w*y - z*x))));
    rpy[2] = std::atan2(2.0f * (w*z + x*y), 1.0f - 2.0f * (y*y + z*z));
    for (int i = 0; i < 3; ++i) rpy[i] *= 180.0f / static_cast<float>(M_PI);
}

void describe(const Scene& scene, const CamCfg& cam, const Options& o) {
    std::printf("scene   %s\n", scene.path.c_str());
    std::printf("        %zu spheres, %zu boxes, %zu cylinders, %zu obbs, ground_plane=%s\n",
                scene.n_spheres(), scene.n_boxes(), scene.n_cylinders(), scene.n_obbs(),
                scene.ground_plane ? "yes" : "no");
    std::printf("camera  %dx%d  fov %.1f°  range [%.2f, %.2f] m  quant %.4f m  pool %d → %dx%d\n",
                cam.width, cam.height, cam.fov_deg, cam.min_range, cam.max_range,
                cam.quantization_m, cam.pool, cam.pooled_h(), cam.pooled_w());
    std::printf("mount   offset [%.3f, %.3f, %.3f] m  rpy [%.1f, %.1f, %.1f]°\n",
                o.mount.offset_body[0], o.mount.offset_body[1], o.mount.offset_body[2],
                o.mount_rpy[0], o.mount_rpy[1], o.mount_rpy[2]);
}

// ---------------------------------------------------------------------------
// Offline rendering (--pose / --poses), used by the parity checker
// ---------------------------------------------------------------------------

int run_offline(const Options& o, const Scene& scene, const CamCfg& cam) {
    std::vector<std::array<float, 7>> poses;

    auto push = [&](const char* line) {
        std::array<float, 7> p{};
        if (std::sscanf(line, "%f %f %f %f %f %f %f",
                        &p[0], &p[1], &p[2], &p[3], &p[4], &p[5], &p[6]) != 7)
            die("pose must be 7 numbers: x y z qw qx qy qz");
        poses.push_back(p);
    };

    if (!o.one_pose.empty()) push(o.one_pose.c_str());
    if (!o.poses_file.empty()) {
        FILE* f = std::fopen(o.poses_file.c_str(), "r");
        if (!f) die("cannot open '" + o.poses_file + "'");
        char line[512];
        while (std::fgets(line, sizeof(line), f)) {
            if (line[0] == '#' || line[0] == '\n') continue;
            push(line);
        }
        std::fclose(f);
    }
    if (poses.empty()) die("--dump needs --pose or --poses");

    RenderBuffers buf;
    buf.resize(cam);

    FILE* dump = nullptr;
    if (!o.dump_path.empty()) {
        dump = std::fopen(o.dump_path.c_str(), "wb");
        if (!dump) die("cannot write '" + o.dump_path + "'");
    }

    for (const auto& p : poses) {
        float body_pos[3]  = {p[0], p[1], p[2]};
        float body_quat[4] = {p[3], p[4], p[5], p[6]};
        float n = std::sqrt(body_quat[0]*body_quat[0] + body_quat[1]*body_quat[1]
                          + body_quat[2]*body_quat[2] + body_quat[3]*body_quat[3]);
        if (!(n > 1e-6f)) die("degenerate quaternion in pose list");
        for (int i = 0; i < 4; ++i) body_quat[i] /= n;

        float cam_pos[3], cam_quat[4];
        apply_mount(o.mount, body_pos, body_quat, cam_pos, cam_quat);

        uint64_t t0 = mono_us();
        render_frame(scene, cam, cam_pos, cam_quat, buf);
        uint64_t dt = mono_us() - t0;

        float mn = buf.raw[0], mx = buf.raw[0];
        for (float v : buf.raw) { mn = std::min(mn, v); mx = std::max(mx, v); }
        std::printf("pose [% .3f % .3f % .3f] q [% .3f % .3f % .3f % .3f]  "
                    "raw [%.3f, %.3f] m  render %llu us\n",
                    p[0], p[1], p[2], body_quat[0], body_quat[1], body_quat[2], body_quat[3],
                    mn, mx, static_cast<unsigned long long>(dt));

        if (dump) {
            std::fwrite(buf.raw.data(),    sizeof(float), buf.raw.size(),    dump);
            std::fwrite(buf.pooled.data(), sizeof(float), buf.pooled.size(), dump);
        }
    }

    if (dump) {
        std::fclose(dump);
        std::printf("wrote %zu frame(s) to %s (per frame: %d raw floats then %d pooled)\n",
                    poses.size(), o.dump_path.c_str(), cam.pixels(), cam.pooled());
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Real-time loop
// ---------------------------------------------------------------------------

void enable_realtime() {
    sched_param sp{};
    sp.sched_priority = 60;
    if (sched_setscheduler(0, SCHED_FIFO, &sp) != 0)
        std::fprintf(stderr, "warning: SCHED_FIFO unavailable (%s) — running at normal priority\n",
                     std::strerror(errno));
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0)
        std::fprintf(stderr, "warning: mlockall failed (%s)\n", std::strerror(errno));
}

}  // namespace

int main(int argc, char** argv) {
    Options o = parse_args(argc, argv);
    set_mount_rpy(o.mount, o.mount_rpy[0], o.mount_rpy[1], o.mount_rpy[2]);

    if (o.sniff > 0) {
        MocapReceiver rx(o.in_addr, o.in_port, o.rb_id);
        std::printf("listening on %s:%u\n", o.in_addr.c_str(), o.in_port);
        rx.sniff(o.sniff, 5000);
        return 0;
    }

    if (o.scene_path.empty()) { usage(); die("--scene is required"); }

    // Scene rotation for the 'n' hotkey.
    std::vector<std::string> scene_files;
    size_t                   scene_index = 0;
    if (!o.scene_dir.empty()) {
        scene_files = list_scene_files(o.scene_dir);
        for (size_t i = 0; i < scene_files.size(); ++i)
            if (scene_files[i] == o.scene_path) scene_index = i;
    }

    Scene  scene;
    CamCfg cam;
    try {
        load_scene(o.scene_path, scene, cam);
    } catch (const std::exception& e) {
        die(e.what());
    }
    apply_overrides(o, cam);

    // The render rate defaults to the camera rate the policy trained with, which
    // export_scene.py copies out of the training config's depth_camera.cam_hz.
    const bool rate_from_scene = (o.rate_hz <= 0.0);
    if (rate_from_scene) o.rate_hz = cam.cam_hz;

    describe(scene, cam, o);
    std::printf("rate    %.2f Hz%s\n", o.rate_hz,
                rate_from_scene ? "  (from the scene file's cam_hz)" : "  (--rate)");

    if (!o.one_pose.empty() || !o.poses_file.empty()) return run_offline(o, scene, cam);

    RenderBuffers buf;
    buf.resize(cam);

    MocapReceiver rx(o.in_addr, o.in_port, o.rb_id);
    DepthSender   tx(o.out_host, o.out_port);
    std::unique_ptr<DepthSender> tx_raw;
    if (o.raw_out) tx_raw.reset(new DepthSender(o.out_host, o.raw_port));

    const std::string rb_note = o.rb_id >= 0
        ? "  (rigid body " + std::to_string(o.rb_id) + ")" : "  (any rigid body)";
    std::printf("pose in  udp %s:%u%s\n", o.in_addr.c_str(), o.in_port, rb_note.c_str());
    std::printf("depth out udp %s:%u  (%d floats/frame)%s\n",
                o.out_host.c_str(), o.out_port, cam.pooled(),
                o.raw_out ? "  + raw frames" : "");
    std::printf("\n");

    if (o.realtime) enable_realtime();

    std::signal(SIGINT,  on_signal);
    std::signal(SIGTERM, on_signal);

    RawTerminal term;
    Preview     preview;
    bool        show_preview = o.preview;
    term.begin();
    if (show_preview) preview.begin();

    const uint64_t period_ns = static_cast<uint64_t>(1e9 / o.rate_hz);

    timespec next;
    clock_gettime(CLOCK_MONOTONIC, &next);

    Pose     pose;                       // last pose received (held when stale)
    uint32_t seq       = 0;
    uint64_t overruns  = 0;
    double   jitter_sum = 0.0;
    double   jitter_max = 0.0;
    double   render_ms  = 0.0;
    uint64_t saved_frames = 0;
    std::string message;

    while (!g_stop) {
        // Absolute deadlines: the period never accumulates drift, so the render
        // rate stays locked to `--rate` regardless of how long a frame takes.
        auto advance = [&] {
            next.tv_nsec += static_cast<long>(period_ns);
            while (next.tv_nsec >= 1000000000L) { next.tv_nsec -= 1000000000L; ++next.tv_sec; }
        };
        advance();

        // A frame that overran its slot would otherwise be chased by a burst of
        // back-to-back renders; skip the missed slots instead and count them.
        timespec now_ts;
        clock_gettime(CLOCK_MONOTONIC, &now_ts);
        while (now_ts.tv_sec > next.tv_sec ||
               (now_ts.tv_sec == next.tv_sec && now_ts.tv_nsec > next.tv_nsec)) {
            advance();
            ++overruns;
        }

        int rc = clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next, nullptr);
        if (rc == EINTR) continue;

        const uint64_t now_us      = mono_us();
        const uint64_t deadline_us = static_cast<uint64_t>(next.tv_sec) * 1000000ull
                                   + next.tv_nsec / 1000;
        const double   jitter      = (static_cast<double>(now_us) - static_cast<double>(deadline_us)) / 1000.0;
        jitter_sum += jitter;
        jitter_max  = std::max(jitter_max, std::fabs(jitter));

        rx.drain(pose);

        // Age must be measured after draining: a packet that arrived during the
        // drain is newer than now_us, and the unsigned difference would wrap.
        const uint64_t drained_us = mono_us();
        const uint64_t pose_age   = (pose.valid && drained_us > pose.recv_us)
                                  ? drained_us - pose.recv_us : 0;
        const bool     stale      = !pose.valid ||
            pose_age > static_cast<uint64_t>(o.pose_timeout_ms) * 1000ull;

        float cam_pos[3], cam_quat[4];
        apply_mount(o.mount, pose.pos, pose.quat, cam_pos, cam_quat);

        const uint64_t t0 = mono_us();
        render_frame(scene, cam, cam_pos, cam_quat, buf);
        const uint64_t t1 = mono_us();
        render_ms = static_cast<double>(t1 - t0) / 1000.0;

        FrameMeta meta;
        meta.seq          = seq++;
        meta.flags        = stale ? FLAG_STALE_POSE : 0;
        meta.pose_time_us = pose.time_us;
        meta.pose_age_us  = static_cast<uint32_t>(std::min<uint64_t>(pose_age, UINT32_MAX));
        meta.render_us    = static_cast<uint32_t>(t1 - t0);
        std::memcpy(meta.pos,  cam_pos,  sizeof(meta.pos));
        std::memcpy(meta.quat, cam_quat, sizeof(meta.quat));

        tx.send_pooled(meta, buf.pooled.data(), cam.pooled_h(), cam.pooled_w());
        if (tx_raw) tx_raw->send_raw_mm(meta, buf.raw.data(), cam.height, cam.width);

        // ---- hotkeys ----
        int key = term.poll_key();
        if (key == 'q') break;
        if (key == 'p') {
            show_preview = !show_preview;
            if (show_preview) preview.begin(); else { preview.end(); std::printf("\n"); }
        }
        if (key == 'r' || key == 'n') {
            std::string path = o.scene_path;
            if (key == 'n' && !scene_files.empty()) {
                scene_index = (scene_index + 1) % scene_files.size();
                path        = scene_files[scene_index];
            }
            try {
                CamCfg new_cam;
                Scene  new_scene;
                load_scene(path, new_scene, new_cam);
                apply_overrides(o, new_cam);
                scene         = std::move(new_scene);
                cam           = new_cam;
                o.scene_path  = path;
                buf.resize(cam);
                message = "loaded " + path;
                // The loop period is fixed at startup; say so rather than
                // quietly rendering at a rate this scene did not ask for.
                if (std::fabs(cam.cam_hz - o.rate_hz) > 1e-3)
                    message += "  [WARNING: its cam_hz is "
                             + std::to_string(cam.cam_hz).substr(0, 5)
                             + " Hz, still rendering at "
                             + std::to_string(o.rate_hz).substr(0, 5)
                             + " Hz — restart to change]";
            } catch (const std::exception& e) {
                message = std::string("scene load failed: ") + e.what();
            }
        }
        if (key == 's') {
            char path[256];
            std::snprintf(path, sizeof(path), "depth_%04llu.pgm",
                          static_cast<unsigned long long>(saved_frames++));
            message = save_pgm(path, cam, buf) ? std::string("saved ") + path
                                               : std::string("failed to save ") + path;
        }

        // ---- status / preview ----
        if (show_preview || (seq % static_cast<uint32_t>(std::max(1.0, o.rate_hz)) == 0)) {
            float rpy[3];
            quat_to_rpy_deg(cam_quat, rpy);

            // A stale frame has three very different causes; naming the right
            // one saves a debugging session.
            std::string why;
            if (stale) {
                if (rx.packets_seen() == 0 && rx.packets_filtered() > 0) {
                    why = "  ← " + std::to_string(rx.packets_filtered())
                        + " packet(s) rejected by --rb-id "
                        + std::to_string(o.rb_id) + "; the wire says id "
                        + std::to_string(rx.last_filtered_id());
                } else if (rx.packets_seen() == 0 && rx.packets_malformed() > 0) {
                    why = "  ← packets arriving but malformed (last size "
                        + std::to_string(rx.last_size()) + " bytes)";
                } else if (rx.packets_seen() == 0) {
                    why = "  ← nothing on udp:" + std::to_string(o.in_port)
                        + " (sender pointed elsewhere, or another process holds the port)";
                } else {
                    why = "  ← stream stopped " + std::to_string(pose_age / 1000) + " ms ago";
                }
            }

            char line[1024];
            int n = std::snprintf(line, sizeof(line),
                "\x1b[1m%s\x1b[0m  %zu prims   %.2f Hz   jitter avg %+.2f ms / max %.2f ms   "
                "overruns %llu\n"
                "render %.3f ms   pose %s (id %u, %.1f ms old)%s\n"
                "rx %llu pkts (%llu filtered, %llu malformed)   tx %llu frames\n"
                "pos [% 7.3f % 7.3f % 7.3f]   rpy [% 6.1f % 6.1f % 6.1f]°%s%s",
                scene.name.c_str(), scene.n_prims(), o.rate_hz,
                seq ? jitter_sum / seq : 0.0, jitter_max,
                static_cast<unsigned long long>(overruns),
                render_ms,
                stale ? "\x1b[31mSTALE\x1b[0m" : "\x1b[32mok\x1b[0m",
                pose.rb_id, static_cast<double>(meta.pose_age_us) / 1000.0, why.c_str(),
                static_cast<unsigned long long>(rx.packets_seen()),
                static_cast<unsigned long long>(rx.packets_filtered()),
                static_cast<unsigned long long>(rx.packets_malformed()),
                static_cast<unsigned long long>(tx.sent()),
                cam_pos[0], cam_pos[1], cam_pos[2], rpy[0], rpy[1], rpy[2],
                message.empty() ? "" : "\n", message.c_str());
            (void)n;

            std::string status(line);
            if (o.print_pooled) {
                status += "\npooled (max-pool → CNN):\n";
                char cell[16];
                for (int j = 0; j < cam.pooled_h(); ++j) {
                    for (int i = 0; i < cam.pooled_w(); ++i) {
                        std::snprintf(cell, sizeof(cell), "%5.2f ",
                                      buf.pooled[static_cast<size_t>(j) * cam.pooled_w() + i]);
                        status += cell;
                    }
                    status += "\n";
                }
            }
            status += "\nkeys: q quit  r reload  n next scene  p preview  s save PGM\n";

            if (show_preview) {
                preview.draw(cam, buf, status);
            } else {
                std::printf("\r%.2f Hz  render %.3f ms  pose %s  tx %llu    ",
                            o.rate_hz, render_ms, stale ? "STALE" : "ok",
                            static_cast<unsigned long long>(tx.sent()));
                std::fflush(stdout);
            }
        }
    }

    preview.end();
    term.end();
    std::printf("\nstopped after %u frames — %llu overrun(s), max jitter %.2f ms, "
                "%llu pose packets, %llu frames sent\n",
                seq, static_cast<unsigned long long>(overruns), jitter_max,
                static_cast<unsigned long long>(rx.packets_seen()),
                static_cast<unsigned long long>(tx.sent()));
    return 0;
}
