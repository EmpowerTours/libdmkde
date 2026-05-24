CXX      ?= g++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -Iinclude

BIN_DIR  := build
BENCH    := $(BIN_DIR)/benchmark
TEST     := $(BIN_DIR)/test_basic

all: $(BENCH) $(TEST)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BENCH): examples/benchmark.cpp include/dmkde.hpp | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $<

$(TEST): tests/test_basic.cpp include/dmkde.hpp | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $<

test: $(TEST)
	@$(TEST) && echo "OK" || (echo "FAILED"; exit 1)

bench: $(BENCH)
	@$(BENCH)

clean:
	rm -rf $(BIN_DIR)

.PHONY: all test bench clean
