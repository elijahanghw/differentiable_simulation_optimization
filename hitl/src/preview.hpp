// preview.hpp — terminal depth preview (ANSI truecolour half-blocks).
//
// Two image rows per text line, so a 64×48 frame occupies 64×24 characters.
// Purely for eyeballing that the mocap pose and the baked scene line up; it is
// drawn after the frame has already been sent, so it cannot delay the output.
#pragma once

#include <string>

#include "render.hpp"
#include "scene.hpp"

class Preview {
public:
    void begin();                       // clear screen, hide cursor
    void end();                         // restore cursor, leave the last frame
    // `status` is printed under the image; embed newlines for extra lines.
    void draw(const CamCfg& cam, const RenderBuffers& buf, const std::string& status);

private:
    std::string out_;                   // reused frame buffer, no per-frame alloc
    bool        started_ = false;
};

// Writes the raw depth frame as a 16-bit PGM (millimetres) — the 's' hotkey.
bool save_pgm(const std::string& path, const CamCfg& cam, const RenderBuffers& buf);
