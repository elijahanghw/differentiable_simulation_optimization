# depth_hitl — ground-station HITL depth renderer.
#
#   make            build ./depth_hitl
#   make native     build tuned for this CPU (do not copy the binary elsewhere)
#   make debug      -O0 -g with sanitizers
#   make clean
#
# No external dependencies: C++17, POSIX sockets, libm.
#
# -ffast-math is deliberately NOT used: the renderer relies on infinities for
# ray misses, and parity with the training-time JAX pipeline matters more than
# the last few percent of speed.

CXX      ?= g++
CC       ?= gcc
CXXFLAGS ?= -std=c++17 -O3 -Wall -Wextra -Wno-unused-parameter
CFLAGS   ?= -std=c99 -O2 -Wall
LDFLAGS  ?=
LDLIBS   ?= -lm

SRC := $(wildcard src/*.cpp)
OBJ := $(SRC:.cpp=.o)
BIN := depth_hitl

# --- CNN encoder (optional) -------------------------------------------------
# `--encode` publishes the policy's 64-float latent instead of the depth tensor.
# Rather than porting the CNN a second time, this compiles in the generated
# cnn_encoder.c — the same translation unit the companion computer flies — so a
# HITL session exercises the deployed encoder itself.
#
# Emitted by indiflight_module/Generate_C_code_navigate_split.ipynb. Detected
# automatically; without it the binary still builds and --encode explains why it
# cannot run. Override with:  make -C hitl ENCODER_DIR=/path/to/onboard
#
# -O2, not -O3: the file is ~8 MB of float literals and the difference is compile
# minutes for a kernel that already runs in a fraction of the 100 ms budget.
ENCODER_DIR ?= ../indiflight_module/c_code_navigate/onboard
ENCODER_SRC := $(wildcard $(ENCODER_DIR)/cnn_encoder.c)

ifneq ($(ENCODER_SRC),)
CXXFLAGS += -DHAVE_CNN_ENCODER -I$(ENCODER_DIR)
ENCODER_OBJ := src/cnn_encoder.o
OBJ += $(ENCODER_OBJ)
endif

.PHONY: all native debug clean

all: $(BIN)
ifeq ($(ENCODER_SRC),)
	@echo "note: no CNN encoder at $(ENCODER_DIR) — built without --encode support"
else
	@echo "note: CNN encoder linked in from $(ENCODER_DIR) — --encode available"
endif

native: CXXFLAGS += -march=native
native: clean $(BIN)

debug: CXXFLAGS := -std=c++17 -O0 -g -Wall -Wextra -Wno-unused-parameter -fsanitize=address,undefined
debug: LDFLAGS  += -fsanitize=address,undefined
debug: clean $(BIN)

$(BIN): $(OBJ)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

src/%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# Generated C, compiled as C and linked in. encoder.cpp includes its header
# inside extern "C", which the generated header also guards for itself.
src/cnn_encoder.o: $(ENCODER_SRC)
	$(CC) $(CFLAGS) -I$(ENCODER_DIR) -c -o $@ $<

src/main.o:    src/scene.hpp src/render.hpp src/mocap.hpp src/netout.hpp src/preview.hpp src/encoder.hpp
src/render.o:  src/render.hpp src/scene.hpp
src/scene.o:   src/scene.hpp src/json.hpp
src/mocap.o:   src/mocap.hpp
src/netout.o:  src/netout.hpp src/mocap.hpp
src/preview.o: src/preview.hpp src/render.hpp src/scene.hpp
src/encoder.o: src/encoder.hpp

clean:
	rm -f src/*.o $(BIN)
