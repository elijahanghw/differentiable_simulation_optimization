#include "scene.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "json.hpp"

namespace {

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open scene file '" + path + "'");
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

const mjson::Array& array_or_empty(const mjson::Value& root, const char* key,
                                   const mjson::Array& empty) {
    if (!root.has(key)) return empty;
    const mjson::Value& v = root.at(key);
    if (v.type != mjson::Value::ARR)
        throw std::runtime_error(std::string("scene: '") + key + "' must be an array");
    return v.arr;
}

void normalize3(float* v) {
    float n = std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    if (n < 1e-9f) throw std::runtime_error("scene: zero-length axis vector");
    v[0] /= n; v[1] /= n; v[2] /= n;
}

}  // namespace

void load_scene(const std::string& path, Scene& scene, CamCfg& cam) {
    mjson::Value root = mjson::parse(read_file(path));
    if (root.type != mjson::Value::OBJ)
        throw std::runtime_error("scene: top level must be an object");

    Scene s;
    s.path         = path;
    s.name         = root.string_or("name", path);
    s.ground_plane = root.bool_or("ground_plane", true);

    const mjson::Array empty;

    for (const mjson::Value& e : array_or_empty(root, "spheres", empty)) {
        float c[3];
        e.at("c").floats(c, 3, "sphere.c");
        s.sphere_c.insert(s.sphere_c.end(), c, c + 3);
        s.sphere_r.push_back(static_cast<float>(e.at("r").number()));
    }

    for (const mjson::Value& e : array_or_empty(root, "boxes", empty)) {
        float c[3], he[3];
        e.at("c").floats(c, 3, "box.c");
        e.at("he").floats(he, 3, "box.he");
        s.box_c.insert(s.box_c.end(), c, c + 3);
        s.box_he.insert(s.box_he.end(), he, he + 3);
    }

    for (const mjson::Value& e : array_or_empty(root, "cylinders", empty)) {
        float c[3], ax[3];
        e.at("c").floats(c, 3, "cylinder.c");
        e.at("ax").floats(ax, 3, "cylinder.ax");
        normalize3(ax);
        s.cyl_c.insert(s.cyl_c.end(), c, c + 3);
        s.cyl_ax.insert(s.cyl_ax.end(), ax, ax + 3);
        s.cyl_hh.push_back(static_cast<float>(e.at("hh").number()));
        s.cyl_r.push_back(static_cast<float>(e.at("r").number()));
    }

    for (const mjson::Value& e : array_or_empty(root, "obbs", empty)) {
        float c[3], q[4], he[3];
        e.at("c").floats(c, 3, "obb.c");
        e.at("q").floats(q, 4, "obb.q");
        e.at("he").floats(he, 3, "obb.he");
        float qn = std::sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
        if (qn < 1e-9f) throw std::runtime_error("scene: zero-length obb quaternion");
        for (int i = 0; i < 4; ++i) q[i] /= qn;
        s.obb_c.insert(s.obb_c.end(), c, c + 3);
        s.obb_q.insert(s.obb_q.end(), q, q + 4);
        s.obb_he.insert(s.obb_he.end(), he, he + 3);
    }

    if (root.has("camera")) {
        const mjson::Value& c = root.at("camera");
        cam.fov_deg        = static_cast<float>(c.number_or("fov_deg",        cam.fov_deg));
        cam.width          = static_cast<int>  (c.number_or("width",          cam.width));
        cam.height         = static_cast<int>  (c.number_or("height",         cam.height));
        cam.min_range      = static_cast<float>(c.number_or("min_range",      cam.min_range));
        cam.max_range      = static_cast<float>(c.number_or("max_range",      cam.max_range));
        cam.quantization_m = static_cast<float>(c.number_or("quantization_m", cam.quantization_m));
        cam.pool           = static_cast<int>  (c.number_or("pool",           cam.pool));
        cam.norm_numerator = static_cast<float>(c.number_or("norm_numerator", cam.norm_numerator));
        cam.norm_clip_min  = static_cast<float>(c.number_or("norm_clip_min",  cam.norm_clip_min));
        cam.norm_offset    = static_cast<float>(c.number_or("norm_offset",    cam.norm_offset));
    }

    scene = std::move(s);
}

std::vector<std::string> list_scene_files(const std::string& dir) {
    std::vector<std::string> out;
    DIR* d = opendir(dir.c_str());
    if (!d) throw std::runtime_error("cannot open scene directory '" + dir + "'");
    while (dirent* e = readdir(d)) {
        std::string name = e->d_name;
        if (name.size() > 5 && name.compare(name.size() - 5, 5, ".json") == 0)
            out.push_back(dir + "/" + name);
    }
    closedir(d);
    std::sort(out.begin(), out.end());
    return out;
}
