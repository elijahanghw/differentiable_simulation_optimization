#include "encoder.hpp"

#ifdef HAVE_CNN_ENCODER
extern "C" {
#include "cnn_encoder.h"
}
#endif

bool Encoder::available() {
#ifdef HAVE_CNN_ENCODER
    return true;
#else
    return false;
#endif
}

std::string Encoder::unavailable_reason() {
#ifdef HAVE_CNN_ENCODER
    return {};
#else
    return "this binary was built without the CNN encoder.\n"
           "  Run indiflight_module/Generate_C_code_navigate_split.ipynb to emit\n"
           "  indiflight_module/c_code_navigate/onboard/cnn_encoder.{c,h}, then\n"
           "  rebuild:  make -C hitl clean && make -C hitl\n"
           "  (point elsewhere with  make -C hitl ENCODER_DIR=/path/to/onboard)";
#endif
}

int Encoder::feature_dim() {
#ifdef HAVE_CNN_ENCODER
    return CNN_FEATURE_DIM;
#else
    return 0;
#endif
}

int Encoder::input_rows() {
#ifdef HAVE_CNN_ENCODER
    return CNN_IN_H;
#else
    return 0;
#endif
}

int Encoder::input_cols() {
#ifdef HAVE_CNN_ENCODER
    return CNN_IN_W;
#else
    return 0;
#endif
}

std::string Encoder::check_shape(int rows, int cols) {
#ifdef HAVE_CNN_ENCODER
    if (rows == CNN_IN_H && cols == CNN_IN_W) return {};
    return "the scene's camera pools to " + std::to_string(rows) + "x" + std::to_string(cols) +
           ", but the encoder was generated for " + std::to_string(CNN_IN_H) + "x" +
           std::to_string(CNN_IN_W) + ".\n"
           "  The checkpoint and the scene file disagree about the camera; regenerate\n"
           "  the scene from the training config the checkpoint came from.";
#else
    (void)rows; (void)cols;
    return unavailable_reason();
#endif
}

void Encoder::run(const float* pooled, float* features) {
#ifdef HAVE_CNN_ENCODER
    // render.cpp writes pooled[pj * PW + pi] and cnn_forward reads
    // depth_in[ph * CNN_IN_W + pw] — same row-major layout, so the tensor goes
    // straight across. cnn_preprocess is deliberately NOT called: render_frame
    // already applied the sensor model, the normalization and the max-pool.
    cnn_forward(pooled, features);
#else
    (void)pooled; (void)features;
#endif
}
