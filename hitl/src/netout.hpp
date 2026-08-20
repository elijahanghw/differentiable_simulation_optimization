// netout.hpp — UDP publisher for rendered depth frames.
//
// One datagram per frame, native little-endian (both ends are LE):
//
//   off  0  char[4]  magic "DPTH"
//   off  4  uint16   version = 1
//   off  6  uint16   flags       bit0: pose is stale / never received
//   off  8  uint32   seq         frame counter, starts at 0
//   off 12  uint16   payload     0 = pooled float32 (CNN input), 1 = raw uint16 mm
//   off 14  uint16   reserved
//   off 16  uint16   rows
//   off 18  uint16   cols
//   off 20  uint32   payload_bytes
//   off 24  uint64   send_time_us    CLOCK_REALTIME on the ground station
//   off 32  uint64   pose_time_us    mocap timestamp this frame was rendered from
//   off 40  uint32   pose_age_us     age of that pose at render time
//   off 44  uint32   render_us       ray-trace cost, for monitoring
//   off 48  float×3  camera position used  (NED)
//   off 60  float×4  camera quaternion used [qw,qx,qy,qz]
//   off 76  payload
//
// Decoder: hitl/tools/recv_depth.py.
#pragma once

#include <cstdint>
#include <netinet/in.h>
#include <string>
#include <vector>

constexpr uint16_t kDepthProtoVersion = 1;
constexpr size_t   kDepthHeaderBytes  = 76;

enum DepthPayload : uint16_t {
    PAYLOAD_POOLED_F32 = 0,
    PAYLOAD_RAW_U16_MM = 1,
};

enum DepthFlags : uint16_t {
    FLAG_STALE_POSE = 1u << 0,
};

struct FrameMeta {
    uint32_t seq          = 0;
    uint16_t flags        = 0;
    uint64_t pose_time_us = 0;
    uint32_t pose_age_us  = 0;
    uint32_t render_us    = 0;
    float    pos[3]       = {0, 0, 0};
    float    quat[4]      = {1, 0, 0, 0};
};

class DepthSender {
public:
    DepthSender(const std::string& host, uint16_t port);
    ~DepthSender();

    DepthSender(const DepthSender&)            = delete;
    DepthSender& operator=(const DepthSender&) = delete;

    // Pooled CNN input, one float per cell.
    bool send_pooled(const FrameMeta& meta, const float* data, int rows, int cols);
    // Full-resolution depth in millimetres, for logging / visualisation.
    bool send_raw_mm(const FrameMeta& meta, const float* metres, int rows, int cols);

    uint64_t sent()   const { return sent_; }
    uint64_t errors() const { return errors_; }

private:
    bool send_buffer(const FrameMeta& meta, uint16_t payload_type, int rows, int cols,
                     const void* payload, size_t payload_bytes);

    int                  fd_ = -1;
    sockaddr_in          dest_{};
    std::vector<uint8_t> buf_;
    std::vector<uint16_t> mm_;
    uint64_t             sent_   = 0;
    uint64_t             errors_ = 0;
};
