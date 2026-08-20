#include "netout.hpp"

#include <arpa/inet.h>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <netdb.h>
#include <stdexcept>
#include <sys/socket.h>
#include <unistd.h>

#include "mocap.hpp"   // wall_us()

namespace {

sockaddr_in resolve(const std::string& host, uint16_t port) {
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(port);
    if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) == 1) return addr;

    addrinfo hints{};
    hints.ai_family   = AF_INET;
    hints.ai_socktype = SOCK_DGRAM;
    addrinfo* res = nullptr;
    if (getaddrinfo(host.c_str(), nullptr, &hints, &res) != 0 || !res)
        throw std::runtime_error("output: cannot resolve host '" + host + "'");
    addr.sin_addr = reinterpret_cast<sockaddr_in*>(res->ai_addr)->sin_addr;
    freeaddrinfo(res);
    return addr;
}

template <typename T>
void put(std::vector<uint8_t>& b, size_t off, T v) {
    std::memcpy(b.data() + off, &v, sizeof(T));
}

}  // namespace

DepthSender::DepthSender(const std::string& host, uint16_t port) {
    dest_ = resolve(host, port);
    fd_   = socket(AF_INET, SOCK_DGRAM, 0);
    if (fd_ < 0) throw std::runtime_error(std::string("output: socket: ") + std::strerror(errno));

    int one = 1;
    setsockopt(fd_, SOL_SOCKET, SO_BROADCAST, &one, sizeof(one));
    buf_.resize(kDepthHeaderBytes);
}

DepthSender::~DepthSender() {
    if (fd_ >= 0) close(fd_);
}

bool DepthSender::send_buffer(const FrameMeta& meta, uint16_t payload_type,
                              int rows, int cols, const void* payload, size_t payload_bytes) {
    buf_.resize(kDepthHeaderBytes + payload_bytes);

    std::memcpy(buf_.data(), "DPTH", 4);
    put<uint16_t>(buf_,  4, kDepthProtoVersion);
    put<uint16_t>(buf_,  6, meta.flags);
    put<uint32_t>(buf_,  8, meta.seq);
    put<uint16_t>(buf_, 12, payload_type);
    put<uint16_t>(buf_, 14, 0);
    put<uint16_t>(buf_, 16, static_cast<uint16_t>(rows));
    put<uint16_t>(buf_, 18, static_cast<uint16_t>(cols));
    put<uint32_t>(buf_, 20, static_cast<uint32_t>(payload_bytes));
    put<uint64_t>(buf_, 24, wall_us());
    put<uint64_t>(buf_, 32, meta.pose_time_us);
    put<uint32_t>(buf_, 40, meta.pose_age_us);
    put<uint32_t>(buf_, 44, meta.render_us);
    for (int i = 0; i < 3; ++i) put<float>(buf_, 48 + 4*i, meta.pos[i]);
    for (int i = 0; i < 4; ++i) put<float>(buf_, 60 + 4*i, meta.quat[i]);
    std::memcpy(buf_.data() + kDepthHeaderBytes, payload, payload_bytes);

    ssize_t n = sendto(fd_, buf_.data(), buf_.size(), 0,
                       reinterpret_cast<sockaddr*>(&dest_), sizeof(dest_));
    if (n != static_cast<ssize_t>(buf_.size())) { ++errors_; return false; }
    ++sent_;
    return true;
}

bool DepthSender::send_pooled(const FrameMeta& meta, const float* data, int rows, int cols) {
    return send_buffer(meta, PAYLOAD_POOLED_F32, rows, cols,
                       data, static_cast<size_t>(rows) * cols * sizeof(float));
}

bool DepthSender::send_raw_mm(const FrameMeta& meta, const float* metres, int rows, int cols) {
    const size_t n = static_cast<size_t>(rows) * cols;
    mm_.resize(n);
    for (size_t i = 0; i < n; ++i) {
        float v = metres[i] * 1000.0f;
        if (!(v > 0.0f)) v = 0.0f;
        if (v > 65535.0f) v = 65535.0f;
        mm_[i] = static_cast<uint16_t>(std::lrint(v));
    }
    return send_buffer(meta, PAYLOAD_RAW_U16_MM, rows, cols, mm_.data(), n * sizeof(uint16_t));
}
