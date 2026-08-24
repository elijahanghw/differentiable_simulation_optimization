#include "mocap.hpp"

#include <arpa/inet.h>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <poll.h>
#include <stdexcept>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

namespace {

constexpr size_t kPacketPadded = 76;   // sender kept pose_t's tail padding
constexpr size_t kPacketPacked = 72;   // sender packed the structs
constexpr size_t kMinPacket    = 40;   // id + timestamp + pos + quat

template <typename T>
T read_le(const uint8_t* p) {
    T v;
    std::memcpy(&v, p, sizeof(T));   // x86 and ARM ground stations are both LE
    return v;
}

}  // namespace

uint64_t mono_us() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000000ull + ts.tv_nsec / 1000;
}

uint64_t wall_us() {
    timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000000ull + ts.tv_nsec / 1000;
}

MocapReceiver::MocapReceiver(const std::string& bind_addr, uint16_t port, int rb_id)
    : rb_filter_(rb_id) {
    fd_ = socket(AF_INET, SOCK_DGRAM, 0);
    if (fd_ < 0) throw std::runtime_error(std::string("mocap: socket: ") + std::strerror(errno));

    int one = 1;
    setsockopt(fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    // The pose stream may be broadcast to the subnet rather than unicast to us.
    setsockopt(fd_, SOL_SOCKET, SO_BROADCAST, &one, sizeof(one));
    int rcvbuf = 1 << 20;
    setsockopt(fd_, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof(rcvbuf));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(port);
    if (bind_addr.empty() || bind_addr == "0.0.0.0") {
        addr.sin_addr.s_addr = INADDR_ANY;
    } else if (inet_pton(AF_INET, bind_addr.c_str(), &addr.sin_addr) != 1) {
        close(fd_);
        throw std::runtime_error("mocap: bad bind address '" + bind_addr + "'");
    }
    if (bind(fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        std::string err = std::strerror(errno);
        close(fd_);
        throw std::runtime_error("mocap: bind " + bind_addr + ":" + std::to_string(port)
                                 + ": " + err);
    }

    int flags = fcntl(fd_, F_GETFL, 0);
    fcntl(fd_, F_SETFL, flags | O_NONBLOCK);
}

MocapReceiver::~MocapReceiver() {
    if (fd_ >= 0)     close(fd_);
    if (fwd_fd_ >= 0) close(fwd_fd_);
}

void MocapReceiver::forward_to(const std::string& host, uint16_t port) {
    fwd_dest_             = sockaddr_in{};
    fwd_dest_.sin_family  = AF_INET;
    fwd_dest_.sin_port    = htons(port);
    if (inet_pton(AF_INET, host.c_str(), &fwd_dest_.sin_addr) != 1)
        throw std::runtime_error("mocap: --pose-forward needs a dotted-quad IPv4 address, "
                                 "got '" + host + "'");

    fwd_fd_ = socket(AF_INET, SOCK_DGRAM, 0);
    if (fwd_fd_ < 0)
        throw std::runtime_error(std::string("mocap: forward socket: ") + std::strerror(errno));

    // The companion computer may be reached over a broadcast address on a small
    // flight-room subnet, same as the inbound stream.
    int one = 1;
    setsockopt(fwd_fd_, SOL_SOCKET, SO_BROADCAST, &one, sizeof(one));
}

int MocapReceiver::wait(int timeout_ms) {
    if (timeout_ms < 0) timeout_ms = 0;
    pollfd pfd{fd_, POLLIN, 0};
    int r = poll(&pfd, 1, timeout_ms);
    return (r > 0 && (pfd.revents & POLLIN)) ? 1 : 0;
}

int MocapReceiver::drain(Pose& latest) {
    uint8_t buf[512];
    int     n_read = 0;

    for (;;) {
        ssize_t n = recv(fd_, buf, sizeof(buf), 0);
        if (n < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) break;
            if (errno == EINTR) continue;
            break;
        }
        last_size_ = static_cast<int>(n);
        if (static_cast<size_t>(n) < kMinPacket) { ++malformed_; continue; }

        uint32_t id = read_le<uint32_t>(buf);
        if (rb_filter_ >= 0 && id != static_cast<uint32_t>(rb_filter_)) {
            ++filtered_;
            last_filtered_id_ = static_cast<long>(id);
            continue;
        }

        Pose p;
        p.rb_id   = id;
        p.time_us = read_le<uint64_t>(buf + 4);
        p.pos[0]  = read_le<float>(buf + 12);
        p.pos[1]  = read_le<float>(buf + 16);
        p.pos[2]  = read_le<float>(buf + 20);

        // Wire order is qx, qy, qz, qw; the renderer wants [qw,qx,qy,qz].
        const float qx = read_le<float>(buf + 24);
        const float qy = read_le<float>(buf + 28);
        const float qz = read_le<float>(buf + 32);
        const float qw = read_le<float>(buf + 36);
        const float qn = std::sqrt(qw*qw + qx*qx + qy*qy + qz*qz);
        if (!(qn > 1e-6f) || !std::isfinite(p.pos[0]) || !std::isfinite(p.pos[1])
            || !std::isfinite(p.pos[2])) {
            ++malformed_;
            continue;
        }
        p.quat[0] = qw / qn; p.quat[1] = qx / qn;
        p.quat[2] = qy / qn; p.quat[3] = qz / qn;

        const size_t der = (static_cast<size_t>(n) >= kPacketPadded) ? 44
                         : (static_cast<size_t>(n) >= kPacketPacked) ? 40
                         : 0;
        if (der && static_cast<size_t>(n) >= der + 32) {
            for (int i = 0; i < 3; ++i) {
                p.vel[i]   = read_le<float>(buf + der + 8  + 4*i);
                p.omega[i] = read_le<float>(buf + der + 20 + 4*i);
            }
        }

        // Relay before keeping: every accepted packet goes on, not just the one
        // this tick renders from, so the companion computer sees the full mocap
        // rate. Sent verbatim — the receiver gets the mocap client's own bytes.
        if (fwd_fd_ >= 0) {
            ssize_t sent = sendto(fwd_fd_, buf, static_cast<size_t>(n), 0,
                                  reinterpret_cast<sockaddr*>(&fwd_dest_), sizeof(fwd_dest_));
            if (sent == n) ++forwarded_; else ++fwd_errors_;
        }

        p.recv_us = mono_us();
        p.valid   = true;
        latest    = p;      // newest packet wins; older ones are simply skipped
        ++packets_;
        ++n_read;
    }
    return n_read;
}

void MocapReceiver::sniff(int count, int timeout_ms) {
    std::printf("sniffing: waiting for %d packet(s)...\n", count);
    uint8_t buf[512];
    for (int got = 0; got < count;) {
        pollfd pfd{fd_, POLLIN, 0};
        int r = poll(&pfd, 1, timeout_ms);
        if (r == 0) { std::printf("  timeout after %d ms\n", timeout_ms); return; }
        if (r < 0) { if (errno == EINTR) continue; std::perror("poll"); return; }

        ssize_t n = recv(fd_, buf, sizeof(buf), 0);
        if (n <= 0) continue;
        ++got;

        std::printf("\npacket %d: %zd bytes", got, n);
        if (n == static_cast<ssize_t>(kPacketPadded))      std::printf("  (matches relay.cpp layout)");
        else if (n == static_cast<ssize_t>(kPacketPacked)) std::printf("  (packed variant)");
        else                                               std::printf("  (UNEXPECTED SIZE)");
        std::printf("\n");

        for (ssize_t i = 0; i < n; i += 16) {
            std::printf("  %04zx  ", i);
            for (ssize_t k = i; k < i + 16; ++k)
                if (k < n) std::printf("%02x ", buf[k]); else std::printf("   ");
            std::printf("\n");
        }
        if (n >= static_cast<ssize_t>(kMinPacket)) {
            const uint32_t id = read_le<uint32_t>(buf);
            // Report the filter verdict here too: a sniff that shows packets
            // while the render loop reports STALE is almost always a wrong
            // --rb-id, and that is invisible unless it is said out loud.
            const char* verdict = (rb_filter_ < 0)              ? "accepted (no --rb-id filter)"
                                : (id == static_cast<uint32_t>(rb_filter_))
                                                                 ? "accepted (matches --rb-id)"
                                                                 : "REJECTED — does not match --rb-id";
            std::printf("  id=%u  t=%llu us   → %s\n", id,
                        static_cast<unsigned long long>(read_le<uint64_t>(buf + 4)), verdict);
            if (rb_filter_ >= 0 && id != static_cast<uint32_t>(rb_filter_))
                std::printf("  hint: rerun with --rb-id %u, or drop the flag to accept any body\n",
                            id);
            std::printf("  pos  = [% .4f, % .4f, % .4f]\n",
                        read_le<float>(buf + 12), read_le<float>(buf + 16), read_le<float>(buf + 20));
            std::printf("  quat = [qw % .4f, qx % .4f, qy % .4f, qz % .4f]\n",
                        read_le<float>(buf + 36), read_le<float>(buf + 24),
                        read_le<float>(buf + 28), read_le<float>(buf + 32));
            if (n >= static_cast<ssize_t>(kPacketPadded)) {
                std::printf("  vel  = [% .4f, % .4f, % .4f]\n",
                            read_le<float>(buf + 52), read_le<float>(buf + 56), read_le<float>(buf + 60));
                std::printf("  rate = [% .4f, % .4f, % .4f]\n",
                            read_le<float>(buf + 64), read_le<float>(buf + 68), read_le<float>(buf + 72));
            }
        }
    }
}
