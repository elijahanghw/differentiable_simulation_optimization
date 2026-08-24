#include "nn_controller.h"
#include <math.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Conventions inherited from training (JADS/tasks/navigate_real.py)
//   world_state[0:3]   position      NED, metres
//   world_state[3:6]   velocity      NED, m/s, world frame
//   world_state[6:9]   roll, pitch, yaw   rad
//   world_state[9:12]  body rates    rad/s
//   world_state[12:16] motor speeds  rad/s (see MOTOR_OMEGA_SCALE)
// ---------------------------------------------------------------------------

#define W_MAX_N     5000.0f   // motor speed mapping to obs = +1 (rad/s)
// Set to (2*M_PI/60) if world_state[12:16] arrives in RPM instead of rad/s.
#define MOTOR_OMEGA_SCALE 1.0f
#define MOTOR_CMD_MAX 1.0f   // 1.0 motor limit, in [-1, 1] units
// Yaw of the training world frame +x axis expressed in the estimator frame.
// Nonzero if the EKF heading origin is not the training origin.
#define WORLD_YAW_OFFSET 0.0f

// motor_order[i] is the indiflight motor carrying training motor i.
// The identified coefficients are per-motor and asymmetric, so a wrong
// order flies badly rather than obviously. Check against the mixer.
static const uint8_t motor_order[4] = {0, 1, 2, 3};

float target_pos[NUM_TARGETS][3] = {
    {3.25f, 0.0f, -1.5f},
};

// Expected arming pose in the training world frame. Not used by
// nn_control(); exported for the indiflight side to position/check against.
const float start_pos[3] = {
    -3.25f, 0.0f, -1.5f
};

const float start_yaw = 0.0f;

const float features_freespace[] = {
    0.0843780637f, 0.0339133553f, 0.0558764637f, 0.0730764046f, -0.0316553824f, -0.165397048f, 0.100379609f, 0.106273383f, 0.0615282543f, 0.071841076f, 0.141774654f, -0.0826975256f,
    0.119216271f, 0.0979460478f, 0.107489929f, 0.0751039684f, -0.165378839f, 0.114046067f, 0.118750453f, 0.18452765f, 0.0729122758f, 0.0415481478f, 0.142602235f, 0.0824027136f,
    0.0444446877f, -0.131468177f, 0.139356062f, 0.00434830785f, -0.13562195f, 0.0367494337f, 0.105318442f, 0.156464458f, -0.0736580044f, 0.185978621f, -0.0309509933f, 0.232680425f,
    0.0158172362f, 0.0505984426f, 0.134052143f, -0.028841598f, 0.087776944f, 0.0358636752f, 0.018315617f, 0.169040352f, -0.0603003204f, 0.0294973142f, 0.148599058f, 0.0870297179f,
    0.105880857f, 0.136784464f, -0.0716602504f, 0.106676102f, 0.250951022f, 0.0254017338f, 0.0960623398f, -0.0631982088f, 0.109255679f, 0.0399420634f, 0.0934086591f, -0.0822480842f,
    -0.20373705f, 0.063033253f, -0.0178487916f, -0.0196989812f
};

static float    nn_hidden[NN_HIDDEN_DIM];
static float    nn_features[NN_FEATURE_DIM];
static uint32_t nn_feature_age  = 0;    // control steps since the last nn_set_features()
static bool     nn_have_feature = false;

uint8_t target_index = 0;

void nn_set_target(uint8_t index, const float p[3])
{
    if (index < NUM_TARGETS) {
        target_pos[index][0] = p[0];
        target_pos[index][1] = p[1];
        target_pos[index][2] = p[2];
    }
}

void nn_reset(void)
{
    target_index = 0;
    nn_reset_hidden(nn_hidden);
    // Seed with the encoder's own output for an empty scene, not with zeros:
    // zeros are not a vector the CNN can produce.
    memcpy(nn_features, features_freespace, sizeof(nn_features));
    nn_feature_age  = 0;
    nn_have_feature = false;
}

// The whole interface to the companion half. Whatever transport delivers the
// vector, this is where it lands; validating it is the caller's job.
void nn_set_features(const float feat[NN_FEATURE_DIM])
{
    memcpy(nn_features, feat, sizeof(nn_features));
    nn_feature_age  = 0;
    nn_have_feature = true;
}

uint32_t nn_features_age(void) { return nn_feature_age; }

int nn_control(const float world_state[16], float motor_cmds[NN_ACT_DIM])
{
    // Advance the waypoint once we are inside the capture radius.
    float dx = world_state[0] - target_pos[target_index][0];
    float dy = world_state[1] - target_pos[target_index][1];
    float dz = world_state[2] - target_pos[target_index][2];
    if (dx*dx + dy*dy + dz*dz < TARGET_RADIUS * TARGET_RADIUS) {
#if TARGET_LOOP
        target_index = (uint8_t)((target_index + 1) % NUM_TARGETS);
#else
        if (target_index + 1 < NUM_TARGETS) { target_index++; }
#endif
        dx = world_state[0] - target_pos[target_index][0];
        dy = world_state[1] - target_pos[target_index][1];
        dz = world_state[2] - target_pos[target_index][2];
    }

    float obs[NN_OBS_DIM];
    // position error, world frame (no target-frame rotation)
    obs[0] = dx;
    obs[1] = dy;
    obs[2] = dz;
    // velocity, world frame
    obs[3] = world_state[3];
    obs[4] = world_state[4];
    obs[5] = world_state[5];
    // attitude
    obs[6] = world_state[6];
    obs[7] = world_state[7];
    float yaw = world_state[8] - WORLD_YAW_OFFSET;
    while (yaw >  (float)M_PI) { yaw -= 2.0f * (float)M_PI; }
    while (yaw < -(float)M_PI) { yaw += 2.0f * (float)M_PI; }
    obs[8] = yaw;
    // body rates
    obs[9]  = world_state[9];
    obs[10] = world_state[10];
    obs[11] = world_state[11];
    // motor speeds scaled to [-1, 1]: w = 2 * W / W_MAX_N - 1
    for (int i = 0; i < NN_ACT_DIM; ++i) {
        float W = world_state[12 + motor_order[i]] * MOTOR_OMEGA_SCALE;
        obs[12 + i] = 2.0f * W / W_MAX_N - 1.0f;
    }

    // Advances nn_hidden. This call is the ONLY writer of the hidden state.
    // nn_features is held between camera frames, which is what training did.
    float action[NN_ACT_DIM];
    nn_forward(nn_features, obs, nn_hidden, action);

    for (int i = 0; i < NN_ACT_DIM; ++i) {
        // clip, then map [-1, 1] -> [0, 1]
        float a = action[i];
        if (a >  MOTOR_CMD_MAX) { a = MOTOR_CMD_MAX; }
        if (a < -1.0f)          { a = -1.0f; }
        motor_cmds[motor_order[i]] = (a + 1.0f) * 0.5f;
    }

    if (nn_feature_age < 0xFFFFFFFFu) { nn_feature_age++; }
    if (!nn_have_feature)                    { return NN_STATUS_NO_FEATURES; }
    if (nn_feature_age > NN_FEATURE_MAX_AGE) { return NN_STATUS_STALE; }
    return NN_STATUS_OK;
}
