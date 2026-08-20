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
CXXFLAGS ?= -std=c++17 -O3 -Wall -Wextra -Wno-unused-parameter
LDFLAGS  ?=
LDLIBS   ?= -lm

SRC := $(wildcard src/*.cpp)
OBJ := $(SRC:.cpp=.o)
BIN := depth_hitl

.PHONY: all native debug clean

all: $(BIN)

native: CXXFLAGS += -march=native
native: clean $(BIN)

debug: CXXFLAGS := -std=c++17 -O0 -g -Wall -Wextra -Wno-unused-parameter -fsanitize=address,undefined
debug: LDFLAGS  += -fsanitize=address,undefined
debug: clean $(BIN)

$(BIN): $(OBJ)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

src/%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

src/main.o:    src/scene.hpp src/render.hpp src/mocap.hpp src/netout.hpp src/preview.hpp
src/render.o:  src/render.hpp src/scene.hpp
src/scene.o:   src/scene.hpp src/json.hpp
src/mocap.o:   src/mocap.hpp
src/netout.o:  src/netout.hpp src/mocap.hpp
src/preview.o: src/preview.hpp src/render.hpp src/scene.hpp

clean:
	rm -f $(OBJ) $(BIN)
