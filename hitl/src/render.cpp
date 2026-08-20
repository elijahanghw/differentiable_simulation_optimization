#include "render.hpp"

#include <cmath>
#include <limits>

namespace {

constexpr float kInf = std::numeric_limits<float>::infinity();

struct Vec3 {
    float x, y, z;
};

inline Vec3 operator-(const Vec3& a, const Vec3& b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline float dot(const Vec3& a, const Vec3& b) { return a.x*b.x + a.y*b.y + a.z*b.z; }

// ---------------------------------------------------------------------------
// Ray–primitive intersections — ports of JADS/depth_render/primitives.py.
// Each returns the hit distance, or +inf on a miss. The 1e-4 epsilons and the
// inside-the-primitive fallbacks are kept exactly as in the JAX version so the
// rendered image matches what the policy saw in training.
// ---------------------------------------------------------------------------

inline float ray_sphere(const Vec3& o, const Vec3& d, const float* c, float r) {
    Vec3  oc   = {o.x - c[0], o.y - c[1], o.z - c[2]};
    float b    = dot(oc, d);
    float cc   = dot(oc, oc) - r * r;
    float disc = b * b - cc;
    if (disc < 0.0f) return kInf;

    float sq  = std::sqrt(disc);
    float t_n = -b - sq;             // front surface
    float t_f = -b + sq;             // back surface (camera inside the sphere)
    if (t_n > 1e-4f) return t_n;
    if (t_f > 1e-4f) return t_f;
    return kInf;
}

// Slab test in a frame where the box spans [-he, +he]; `o`/`d` are already local.
inline float ray_aabb_local(const Vec3& o, const Vec3& d, const float* he) {
    const float od[3] = {o.x, o.y, o.z};
    const float dd[3] = {d.x, d.y, d.z};

    float t_enter = -kInf;
    float t_exit  =  kInf;
    for (int a = 0; a < 3; ++a) {
        // jnp.where(|d| > 1e-10, d, sign(d + 1e-30) * 1e-10): d == 0 → +1e-10.
        float safe_d = std::fabs(dd[a]) > 1e-10f ? dd[a] : (dd[a] < 0.0f ? -1e-10f : 1e-10f);
        float inv    = 1.0f / safe_d;
        float t1     = (-he[a] - od[a]) * inv;
        float t2     = ( he[a] - od[a]) * inv;
        float lo     = t1 < t2 ? t1 : t2;
        float hi     = t1 < t2 ? t2 : t1;
        if (lo > t_enter) t_enter = lo;
        if (hi < t_exit)  t_exit  = hi;
    }

    if (!(t_exit >= t_enter && t_exit > 1e-4f)) return kInf;
    return t_enter > 1e-4f ? t_enter : t_exit;
}

inline void quat_rot_transpose(const float* q, const Vec3& v, Vec3& out) {
    // R^T @ v, with R the rotation matrix of [qw,qx,qy,qz].
    const float qw = q[0], qx = q[1], qy = q[2], qz = q[3];
    const float r00 = 1 - 2*(qy*qy + qz*qz), r01 = 2*(qx*qy - qz*qw), r02 = 2*(qx*qz + qy*qw);
    const float r10 = 2*(qx*qy + qz*qw), r11 = 1 - 2*(qx*qx + qz*qz), r12 = 2*(qy*qz - qx*qw);
    const float r20 = 2*(qx*qz - qy*qw), r21 = 2*(qy*qz + qx*qw), r22 = 1 - 2*(qx*qx + qy*qy);
    out.x = r00*v.x + r10*v.y + r20*v.z;
    out.y = r01*v.x + r11*v.y + r21*v.z;
    out.z = r02*v.x + r12*v.y + r22*v.z;
}

inline float ray_obb(const Vec3& o, const Vec3& d, const float* c, const float* q, const float* he) {
    Vec3 rel = {o.x - c[0], o.y - c[1], o.z - c[2]};
    Vec3 lo, ld;
    quat_rot_transpose(q, rel, lo);
    quat_rot_transpose(q, d,   ld);
    return ray_aabb_local(lo, ld, he);
}

// Finite open cylinder — lateral surface only. The end caps arrive as spheres.
inline float ray_cylinder(const Vec3& o, const Vec3& d, const float* c,
                          const float* ax, float half_h, float r) {
    Vec3 axis = {ax[0], ax[1], ax[2]};
    Vec3 A    = {c[0] - half_h*axis.x, c[1] - half_h*axis.y, c[2] - half_h*axis.z};
    Vec3 AO   = o - A;

    float d_along  = dot(d, axis);
    float ao_along = dot(AO, axis);

    float a_q    = 1.0f - d_along * d_along;
    float b_half = dot(d, AO) - d_along * ao_along;
    float c_q    = dot(AO, AO) - ao_along * ao_along - r * r;
    float disc   = b_half * b_half - a_q * c_q;

    if (disc < 0.0f || a_q <= 1e-12f) return kInf;

    float sq      = std::sqrt(disc);
    float inv_len = 0.5f / half_h;
    float best    = kInf;
    for (int s = 0; s < 2; ++s) {
        float t = (s == 0 ? (-b_half - sq) : (-b_half + sq)) / a_q;
        if (t <= 1e-4f) continue;
        float proj = (ao_along + t * d_along) * inv_len;   // 0..1 along the segment
        if (proj < 0.0f || proj > 1.0f) continue;
        if (t < best) best = t;
    }
    return best;
}

// Ground plane at z = 0 with normal [0,0,-1] (NED: up).
inline float ray_ground(const Vec3& o, const Vec3& d) {
    float denom = -d.z;
    if (std::fabs(denom) <= 1e-6f) return kInf;
    float t = o.z / denom;             // dot(point - o, n) / denom, point = origin
    return t > 1e-4f ? t : kInf;
}

}  // namespace

// ---------------------------------------------------------------------------
// Camera mount
// ---------------------------------------------------------------------------

void set_mount_rpy(CamMount& mount, float roll_deg, float pitch_deg, float yaw_deg) {
    const float k  = static_cast<float>(M_PI) / 180.0f;
    const float cr = std::cos(roll_deg  * k * 0.5f), sr = std::sin(roll_deg  * k * 0.5f);
    const float cp = std::cos(pitch_deg * k * 0.5f), sp = std::sin(pitch_deg * k * 0.5f);
    const float cy = std::cos(yaw_deg   * k * 0.5f), sy = std::sin(yaw_deg   * k * 0.5f);

    // Matches JADS/drone_physics/quat_math.euler_to_quat.
    mount.quat[0] = cr*cp*cy + sr*sp*sy;
    mount.quat[1] = sr*cp*cy - cr*sp*sy;
    mount.quat[2] = cr*sp*cy + sr*cp*sy;
    mount.quat[3] = cr*cp*sy - sr*sp*cy;

    mount.identity = (roll_deg == 0.0f && pitch_deg == 0.0f && yaw_deg == 0.0f)
                     && mount.offset_body[0] == 0.0f
                     && mount.offset_body[1] == 0.0f
                     && mount.offset_body[2] == 0.0f;
}

void apply_mount(const CamMount& mount,
                 const float body_pos[3], const float body_quat[4],
                 float cam_pos[3], float cam_quat[4]) {
    const float qw = body_quat[0], qx = body_quat[1], qy = body_quat[2], qz = body_quat[3];

    // World position of the camera: p + R_body @ offset_body.
    const float r00 = 1 - 2*(qy*qy + qz*qz), r01 = 2*(qx*qy - qz*qw), r02 = 2*(qx*qz + qy*qw);
    const float r10 = 2*(qx*qy + qz*qw), r11 = 1 - 2*(qx*qx + qz*qz), r12 = 2*(qy*qz - qx*qw);
    const float r20 = 2*(qx*qz - qy*qw), r21 = 2*(qy*qz + qx*qw), r22 = 1 - 2*(qx*qx + qy*qy);
    const float* o  = mount.offset_body;
    cam_pos[0] = body_pos[0] + r00*o[0] + r01*o[1] + r02*o[2];
    cam_pos[1] = body_pos[1] + r10*o[0] + r11*o[1] + r12*o[2];
    cam_pos[2] = body_pos[2] + r20*o[0] + r21*o[1] + r22*o[2];

    // Camera orientation: q_body ⊗ q_mount (mount is expressed in the body frame).
    const float mw = mount.quat[0], mx = mount.quat[1], my = mount.quat[2], mz = mount.quat[3];
    cam_quat[0] = qw*mw - qx*mx - qy*my - qz*mz;
    cam_quat[1] = qw*mx + qx*mw + qy*mz - qz*my;
    cam_quat[2] = qw*my - qx*mz + qy*mw + qz*mx;
    cam_quat[3] = qw*mz + qx*my - qy*mx + qz*mw;

    float n = std::sqrt(cam_quat[0]*cam_quat[0] + cam_quat[1]*cam_quat[1]
                      + cam_quat[2]*cam_quat[2] + cam_quat[3]*cam_quat[3]);
    if (n > 1e-9f) for (int i = 0; i < 4; ++i) cam_quat[i] /= n;
}

// ---------------------------------------------------------------------------
// Frame
// ---------------------------------------------------------------------------

void render_frame(const Scene& scene, const CamCfg& cam,
                  const float cam_pos[3], const float cam_quat[4],
                  RenderBuffers& buf) {
    const float qw = cam_quat[0], qx = cam_quat[1], qy = cam_quat[2], qz = cam_quat[3];

    // Camera basis (camera.py::_quat_to_basis): forward = R@+X, right = R@+Y,
    // cam_up = R@−Z.
    const Vec3 forward = {1 - 2*(qy*qy + qz*qz), 2*(qx*qy + qz*qw), 2*(qx*qz - qy*qw)};
    const Vec3 right   = {2*(qx*qy - qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz + qx*qw)};
    const Vec3 cam_up  = {-(2*(qx*qz + qy*qw)), -(2*(qy*qz - qx*qw)), -(1 - 2*(qx*qx + qy*qy))};

    const int   W = cam.width, H = cam.height;
    const float aspect = static_cast<float>(W) / static_cast<float>(H);
    const float tan_h  = std::tan(cam.fov_deg * static_cast<float>(M_PI) / 180.0f * 0.5f);
    const float tan_v  = tan_h / aspect;

    const Vec3 origin = {cam_pos[0], cam_pos[1], cam_pos[2]};

    const size_t n_sph = scene.n_spheres();
    const size_t n_box = scene.n_boxes();
    const size_t n_cyl = scene.n_cylinders();
    const size_t n_obb = scene.n_obbs();

    for (int j = 0; j < H; ++j) {
        // v is negated so row 0 is the top of the image.
        const float v = -(((static_cast<float>(j) + 0.5f) / static_cast<float>(H)) * 2.0f - 1.0f);
        for (int i = 0; i < W; ++i) {
            const float u = ((static_cast<float>(i) + 0.5f) / static_cast<float>(W)) * 2.0f - 1.0f;

            Vec3 d = {u * tan_h * right.x + v * tan_v * cam_up.x + forward.x,
                      u * tan_h * right.y + v * tan_v * cam_up.y + forward.y,
                      u * tan_h * right.z + v * tan_v * cam_up.z + forward.z};
            const float inv_n = 1.0f / std::sqrt(dot(d, d));
            d.x *= inv_n; d.y *= inv_n; d.z *= inv_n;

            float t = scene.ground_plane ? ray_ground(origin, d) : kInf;

            for (size_t k = 0; k < n_sph; ++k) {
                float h = ray_sphere(origin, d, &scene.sphere_c[3*k], scene.sphere_r[k]);
                if (h < t) t = h;
            }
            for (size_t k = 0; k < n_box; ++k) {
                const float* c = &scene.box_c[3*k];
                Vec3 lo = {origin.x - c[0], origin.y - c[1], origin.z - c[2]};
                float h = ray_aabb_local(lo, d, &scene.box_he[3*k]);
                if (h < t) t = h;
            }
            for (size_t k = 0; k < n_cyl; ++k) {
                float h = ray_cylinder(origin, d, &scene.cyl_c[3*k], &scene.cyl_ax[3*k],
                                       scene.cyl_hh[k], scene.cyl_r[k]);
                if (h < t) t = h;
            }
            for (size_t k = 0; k < n_obb; ++k) {
                float h = ray_obb(origin, d, &scene.obb_c[3*k], &scene.obb_q[4*k],
                                  &scene.obb_he[3*k]);
                if (h < t) t = h;
            }

            // apply_sensor_noise: saturate, blind zone, quantize.
            float dep = std::isfinite(t) ? t : cam.max_range;
            if (dep > cam.max_range) dep = cam.max_range;
            if (dep < cam.min_range) dep = 0.0f;
            if (cam.quantization_m > 0.0f && dep > 0.0f) {
                // rintf is round-half-to-even under the default rounding mode,
                // matching jnp.round.
                dep = std::rint(dep / cam.quantization_m) * cam.quantization_m;
            }

            const size_t idx = static_cast<size_t>(j) * W + i;
            buf.raw[idx] = dep;

            float clipped = dep;
            if (clipped < cam.norm_clip_min) clipped = cam.norm_clip_min;
            if (clipped > cam.max_range)     clipped = cam.max_range;
            buf.normed[idx] = cam.norm_numerator / clipped - cam.norm_offset;
        }
    }

    // 4×4 max-pool, stride 4, VALID padding → CNN input.
    const int P  = cam.pool;
    const int PH = cam.pooled_h(), PW = cam.pooled_w();
    for (int pj = 0; pj < PH; ++pj) {
        for (int pi = 0; pi < PW; ++pi) {
            float m = -kInf;
            for (int dj = 0; dj < P; ++dj) {
                const float* row = &buf.normed[static_cast<size_t>(pj*P + dj) * W + pi*P];
                for (int di = 0; di < P; ++di)
                    if (row[di] > m) m = row[di];
            }
            buf.pooled[static_cast<size_t>(pj) * PW + pi] = m;
        }
    }
}
