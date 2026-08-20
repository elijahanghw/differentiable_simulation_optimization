#include "preview.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <sys/ioctl.h>
#include <unistd.h>

namespace {

struct Rgb { int r, g, b; };

// Near → far: red, orange, yellow, green, blue. The blind zone (0 m) and the
// saturated far plane get their own flat colours so they read as "no data".
Rgb depth_colour(float d, const CamCfg& cam) {
    if (d <= 0.0f)                 return {200,  0, 200};   // closer than min_range
    if (d >= cam.max_range)        return { 20, 20,  30};   // no hit / saturated

    float t = (d - cam.min_range) / (cam.max_range - cam.min_range);
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;

    static const Rgb stops[5] = {{255, 40, 40}, {255, 160, 0}, {245, 245, 0},
                                 {40, 220, 60}, {40, 90, 255}};
    float  f = t * 4.0f;
    int    i = static_cast<int>(f);
    if (i > 3) i = 3;
    float  a = f - static_cast<float>(i);
    const Rgb& c0 = stops[i];
    const Rgb& c1 = stops[i + 1];
    return {static_cast<int>(c0.r + a * (c1.r - c0.r)),
            static_cast<int>(c0.g + a * (c1.g - c0.g)),
            static_cast<int>(c0.b + a * (c1.b - c0.b))};
}

int terminal_cols() {
    winsize ws{};
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0) return ws.ws_col;
    return 80;
}

void append_fg_bg(std::string& s, const Rgb& fg, const Rgb& bg) {
    char tmp[64];
    std::snprintf(tmp, sizeof(tmp), "\x1b[38;2;%d;%d;%d;48;2;%d;%d;%dm",
                  fg.r, fg.g, fg.b, bg.r, bg.g, bg.b);
    s += tmp;
}

}  // namespace

void Preview::begin() {
    started_ = true;
    std::fputs("\x1b[2J\x1b[?25l", stdout);   // clear, hide cursor
    std::fflush(stdout);
}

void Preview::end() {
    if (!started_) return;
    std::fputs("\x1b[0m\x1b[?25h\n", stdout);  // reset colours, show cursor
    std::fflush(stdout);
    started_ = false;
}

void Preview::draw(const CamCfg& cam, const RenderBuffers& buf, const std::string& status) {
    const int W = cam.width, H = cam.height;
    // Subsample if the frame is wider than the terminal.
    int step = 1;
    while (W / step > terminal_cols() - 1) ++step;

    out_.clear();
    out_ += "\x1b[H";                       // cursor home — no full clear, less flicker

    for (int j = 0; j + step < H; j += 2 * step) {
        Rgb prev_fg{-1, -1, -1}, prev_bg{-1, -1, -1};
        for (int i = 0; i < W; i += step) {
            Rgb top = depth_colour(buf.raw[static_cast<size_t>(j) * W + i], cam);
            Rgb bot = depth_colour(buf.raw[static_cast<size_t>(j + step) * W + i], cam);
            if (top.r != prev_fg.r || top.g != prev_fg.g || top.b != prev_fg.b ||
                bot.r != prev_bg.r || bot.g != prev_bg.g || bot.b != prev_bg.b) {
                append_fg_bg(out_, top, bot);
                prev_fg = top;
                prev_bg = bot;
            }
            out_ += "▀";               // ▀ upper half block
        }
        out_ += "\x1b[0m\x1b[K\n";
    }

    out_ += "\x1b[0m\x1b[K\n";
    out_ += status;
    out_ += "\x1b[K";
    // Clear anything left over from a previously longer status block.
    out_ += "\x1b[J";

    ssize_t ignored = write(STDOUT_FILENO, out_.data(), out_.size());
    (void)ignored;
}

bool save_pgm(const std::string& path, const CamCfg& cam, const RenderBuffers& buf) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    std::fprintf(f, "P5\n%d %d\n65535\n", cam.width, cam.height);
    for (int i = 0; i < cam.pixels(); ++i) {
        float v = buf.raw[i] * 1000.0f;              // millimetres
        if (!(v > 0.0f)) v = 0.0f;
        if (v > 65535.0f) v = 65535.0f;
        unsigned mm = static_cast<unsigned>(std::lrint(v));
        unsigned char be[2] = {static_cast<unsigned char>(mm >> 8),
                               static_cast<unsigned char>(mm & 0xFF)};   // PGM is big-endian
        std::fwrite(be, 1, 2, f);
    }
    std::fclose(f);
    return true;
}
