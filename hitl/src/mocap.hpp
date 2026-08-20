// mocap.hpp — receiver for the OptiTrack pose datagram.
//
// Wire format is the one relay.cpp consumes (tmp/relay.cpp, OPTITRACK_PORT),
// produced by the unified-mocap-client agent. Native little-endian, and it
// carries the tail padding of the sender's pose_t struct:
//
//   off  0  uint32   rigid-body streaming id
//   off  4  uint64   pose timestamp (µs, camera mid-exposure)
//   off 12  float×3  position   x, y, z          NED, metres
//   off 24  float×4  quaternion qx, qy, qz, qw   body→world
//   off 40  (4 bytes struct padding)
//   off 44  uint64   derivative timestamp (µs)
//   off 52  float×3  velocity   x, y, z          NED, m/s
//   off 64  float×3  body rates wx, wy, wz       rad/s
//   = 76 bytes
//
// A 72-byte variant (sender built without the padding) is accepted too; the
// pose fields sit at the same offsets either way.
#pragma once

#include <cstdint>
#include <string>

struct Pose {
    uint32_t rb_id    = 0;
    uint64_t time_us  = 0;      // sender's timestamp, echoed downstream
    uint64_t recv_us  = 0;      // CLOCK_MONOTONIC arrival time, for staleness
    float    pos[3]   = {0, 0, 0};
    float    quat[4]  = {1, 0, 0, 0};   // [qw,qx,qy,qz], normalized
    float    vel[3]   = {0, 0, 0};
    float    omega[3] = {0, 0, 0};
    bool     valid    = false;
};

class MocapReceiver {
public:
    // bind_addr "0.0.0.0" listens on every interface. rb_id < 0 accepts any
    // rigid body; otherwise only that streaming id is used.
    MocapReceiver(const std::string& bind_addr, uint16_t port, int rb_id);
    ~MocapReceiver();

    MocapReceiver(const MocapReceiver&)            = delete;
    MocapReceiver& operator=(const MocapReceiver&) = delete;

    // Drains the socket and keeps the newest acceptable packet. Returns the
    // number of packets read. Never blocks.
    int drain(Pose& latest);

    // Blocking-ish hexdump of the next `count` packets, for wire debugging.
    void sniff(int count, int timeout_ms);

    uint64_t packets_seen()    const { return packets_; }
    uint64_t packets_dropped() const { return dropped_; }
    int      last_size()       const { return last_size_; }

private:
    int      fd_        = -1;
    int      rb_filter_ = -1;
    uint64_t packets_   = 0;
    uint64_t dropped_   = 0;   // wrong size, or filtered out by rigid-body id
    int      last_size_ = 0;
};

// CLOCK_MONOTONIC in microseconds — one place so every timestamp agrees.
uint64_t mono_us();
// CLOCK_REALTIME in microseconds — for stamps that leave this machine.
uint64_t wall_us();
