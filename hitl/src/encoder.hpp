// encoder.hpp — the policy's CNN encoder, running on the ground station.
//
// The HITL renderer normally publishes the 12×16 depth tensor and leaves the CNN
// to whatever consumes the stream. With --encode it runs the encoder here instead
// and publishes the 64-float latent, which is what the split deployment actually
// puts on the wire (see indiflight_module/Generate_C_code_navigate_split.ipynb).
//
// This compiles the *generated* cnn_encoder.c — the same translation unit the
// companion computer flies — rather than a second port of it. So a HITL flight
// exercises the deployed encoder, not a reimplementation that could drift from it.
// The Makefile finds it automatically; without it the binary still builds and
// --encode reports why it cannot run.
#pragma once

#include <string>

class Encoder {
public:
    // True if the binary was built with the generated encoder linked in.
    static bool available();
    // Why not, for the error message. Empty when available().
    static std::string unavailable_reason();

    // Feature vector width, or 0 when unavailable.
    static int feature_dim();
    // Depth tensor the encoder expects, so main() can check it against the camera.
    static int input_rows();
    static int input_cols();

    // Checks that a pooled tensor of `rows`×`cols` is what the encoder was
    // generated for. Returns an explanatory message on mismatch, empty on success.
    static std::string check_shape(int rows, int cols);

    // One frame: normalized, pooled depth (row-major, rows*cols) → features.
    // `features` must have room for feature_dim(). Stateless.
    static void run(const float* pooled, float* features);
};
