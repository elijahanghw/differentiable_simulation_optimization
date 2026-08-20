// scene.hpp — static obstacle set + camera parameters for one HITL episode.
//
// Loaded from the JSON written by hitl/tools/export_scene.py, which samples it
// with the same JADS/scene/scene.py SceneConfig the policy trained against.
// Geometry layout mirrors JADS/depth_render/renderer.py exactly:
//   spheres   — includes the end-cap spheres of every capsule
//   boxes     — axis-aligned (AABB)
//   cylinders — open lateral surface only; the caps are the spheres above
//   obbs      — oriented boxes (tree branches)
// A ground plane at z = 0 (NED, normal [0,0,-1]) is implicit unless disabled.
#pragma once

#include <string>
#include <vector>

struct CamCfg {
    float fov_deg        = 80.0f;   // horizontal FOV
    int   width          = 64;
    int   height         = 48;
    float min_range      = 0.3f;    // closer than this → 0 (blind zone)
    float max_range      = 3.0f;    // farther / no hit → max_range
    float quantization_m = 0.001f;  // depth step (1 mm)
    float cam_hz         = 10.0f;   // rate the policy was trained to expect
    int   pool           = 4;       // max-pool factor → CNN input

    // Normalization applied before pooling, hardcoded in NavigateReal
    // ._get_processed_depth as  3.0 / clip(raw, 0.3, max_range) - 0.6.
    float norm_numerator = 3.0f;
    float norm_clip_min  = 0.3f;
    float norm_offset    = 0.6f;

    int pooled_h() const { return height / pool; }
    int pooled_w() const { return width  / pool; }
    int pixels()   const { return width * height; }
    int pooled()   const { return pooled_h() * pooled_w(); }
};

struct Scene {
    // Spheres
    std::vector<float> sphere_c;   // 3 per sphere
    std::vector<float> sphere_r;

    // Axis-aligned boxes
    std::vector<float> box_c;      // 3 per box
    std::vector<float> box_he;     // 3 per box

    // Open cylinders (capsule bodies)
    std::vector<float> cyl_c;      // 3 per cylinder
    std::vector<float> cyl_ax;     // 3 per cylinder (unit axis)
    std::vector<float> cyl_hh;
    std::vector<float> cyl_r;

    // Oriented boxes
    std::vector<float> obb_c;      // 3 per obb
    std::vector<float> obb_q;      // 4 per obb, [qw,qx,qy,qz]
    std::vector<float> obb_he;     // 3 per obb

    bool        ground_plane = true;
    std::string name;              // for the status line
    std::string path;

    size_t n_spheres()   const { return sphere_r.size(); }
    size_t n_boxes()     const { return box_c.size() / 3; }
    size_t n_cylinders() const { return cyl_r.size(); }
    size_t n_obbs()      const { return obb_q.size() / 4; }
    size_t n_prims()     const { return n_spheres() + n_boxes() + n_cylinders() + n_obbs(); }
};

// Loads the scene, and any "camera" block it carries, into `cam`.
// Throws std::runtime_error on a malformed or missing file.
void load_scene(const std::string& path, Scene& scene, CamCfg& cam);

// Sorted list of *.json under `dir` — backs the "next scene" hotkey.
std::vector<std::string> list_scene_files(const std::string& dir);
