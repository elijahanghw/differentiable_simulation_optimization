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
#include <netinet/in.h>
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

    // Relay every accepted pose datagram, byte for byte, to another host — for
    // when the mocap client can only unicast to one destination and the drone's
    // companion computer needs the stream too.
    //
    // Forwarding happens inside drain(), per packet, so the destination gets the
    // full mocap rate. Publishing the pose alongside the depth frames instead
    // would cap it at the render rate, which is an order of magnitude too slow
    // for a 100 Hz controller.
    //
    // Only packets that pass the size, --rb-id and finiteness checks are
    // relayed, and they go out unmodified: the receiver sees exactly the bytes
    // the mocap client sent, timestamps and 72/76-byte variant included.
    void forward_to(const std::string& host, uint16_t port);

    // Drains the socket and keeps the newest acceptable packet. Returns the
    // number of packets read. Never blocks.
    int drain(Pose& latest);

    // Blocks for up to timeout_ms waiting for a packet to arrive. >0 if one is
    // ready. Lets the caller relay poses as they land rather than once per
    // render tick — see forward_to().
    int wait(int timeout_ms);

    // Blocking-ish hexdump of the next `count` packets, for wire debugging.
    void sniff(int count, int timeout_ms);

    uint64_t packets_seen()      const { return packets_; }
    uint64_t packets_dropped()   const { return malformed_ + filtered_; }
    uint64_t packets_filtered()  const { return filtered_; }   // wrong rigid-body id
    uint64_t packets_malformed() const { return malformed_; }  // bad size / not finite
    int      last_size()         const { return last_size_; }
    // Id of the most recent packet rejected by the --rb-id filter, so a
    // mismatch reports the id that is actually on the wire.
    long     last_filtered_id()  const { return last_filtered_id_; }

    bool     forwarding()        const { return fwd_fd_ >= 0; }
    uint64_t forwarded()         const { return forwarded_; }
    uint64_t forward_errors()    const { return fwd_errors_; }

private:
    int         fd_              = -1;
    int         rb_filter_       = -1;
    uint64_t    packets_         = 0;
    uint64_t    filtered_        = 0;
    uint64_t    malformed_       = 0;
    int         last_size_       = 0;
    long        last_filtered_id_ = -1;

    int         fwd_fd_          = -1;
    sockaddr_in fwd_dest_{};
    uint64_t    forwarded_       = 0;
    uint64_t    fwd_errors_      = 0;
};

// CLOCK_MONOTONIC in microseconds — one place so every timestamp agrees.
uint64_t mono_us();
// CLOCK_REALTIME in microseconds — for stamps that leave this machine.
uint64_t wall_us();
