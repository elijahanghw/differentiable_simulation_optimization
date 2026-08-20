// render.hpp — CPU depth renderer, a line-for-line port of the JAX pipeline in
// JADS/depth_render/{camera,primitives,renderer}.py plus the post-processing in
// NavigateReal._get_processed_depth.
//
//   generate rays → min over primitives → apply_sensor_noise
//                 → 3/clip(d,0.3,max) - 0.6 → 4×4 max-pool → CNN input
//
// Conventions (NED): X forward, Y right, Z down. Quaternion is [qw,qx,qy,qz],
// body→world. Camera looks along body +X, image "up" is body −Z.
#pragma once

#include <cstddef>

#include "scene.hpp"

// Per-frame scratch + outputs. Allocated once; render() never allocates.
struct RenderBuffers {
    std::vector<float> raw;      // height*width — metres, after sensor noise model
    std::vector<float> normed;   // height*width — 3/clip(raw,0.3,max) - 0.6
    std::vector<float> pooled;   // pooled_h*pooled_w — CNN input

    void resize(const CamCfg& cam) {
        raw.assign(static_cast<size_t>(cam.pixels()), 0.0f);
        normed.assign(static_cast<size_t>(cam.pixels()), 0.0f);
        pooled.assign(static_cast<size_t>(cam.pooled()), 0.0f);
    }
};

// Camera mount: fixed transform from the mocap rigid-body frame to the camera.
// Defaults are the exact sim convention (camera at the body origin, body-aligned).
struct CamMount {
    float offset_body[3] = {0.0f, 0.0f, 0.0f};       // metres, body frame (FRD)
    float quat[4]        = {1.0f, 0.0f, 0.0f, 0.0f}; // [qw,qx,qy,qz], body→camera
    bool  identity       = true;                      // set by set_mount_rpy()
};

// Fills mount.quat from roll/pitch/yaw in degrees (ZYX intrinsic, same
// convention as JADS/drone_physics/quat_math.euler_to_quat).
void set_mount_rpy(CamMount& mount, float roll_deg, float pitch_deg, float yaw_deg);

// Applies the mount transform: body pose → camera pose.
void apply_mount(const CamMount& mount,
                 const float body_pos[3], const float body_quat[4],
                 float cam_pos[3], float cam_quat[4]);

// Full pipeline for one frame. `cam_quat` must be normalized.
// Fills buf.raw, buf.normed and buf.pooled.
void render_frame(const Scene& scene, const CamCfg& cam,
                  const float cam_pos[3], const float cam_quat[4],
                  RenderBuffers& buf);
