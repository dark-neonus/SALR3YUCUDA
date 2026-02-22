# ═══════════════════════════════════════════════════════════════════════════════
#  SALR Particle Simulation — Makefile  (pure C, CPU-only for now)
# ═══════════════════════════════════════════════════════════════════════════════
#
#  Targets:
#    make all             — build main executable + tests
#    make salr_dft        — build main executable
#    make tests           — build all test programs
#    make clean           — remove build artifacts
#
# ═══════════════════════════════════════════════════════════════════════════════

# ── Compiler settings ─────────────────────────────────────────────────────────
CC       = gcc
CFLAGS   = -O2 -Wall -Wextra -std=c11

# ── Directories ───────────────────────────────────────────────────────────────
BUILD_DIR = build
INCLUDE   = include
SRC_CORE  = src/core
SRC_CPU   = src/cpu
SRC_UTILS = src/utils
TESTS     = tests

# Ensure build directory exists
$(shell mkdir -p $(BUILD_DIR))

# ── Source files ──────────────────────────────────────────────────────────────
MAIN_SRCS = src/main.c \
            $(SRC_CORE)/config.c \
            $(SRC_CORE)/grid.c \
            $(SRC_CPU)/potential_cpu.c \
            $(SRC_CPU)/solver_cpu.c \
            $(SRC_CPU)/math_utils_cpu.c \
            $(SRC_UTILS)/io.c

# ═══════════════════════════════════════════════════════════════════════════════
#  TARGETS
# ═══════════════════════════════════════════════════════════════════════════════

.PHONY: all tests clean

all: salr_dft tests

# ── Main executable ───────────────────────────────────────────────────────────
salr_dft: $(MAIN_SRCS)
	$(CC) $(CFLAGS) -I$(INCLUDE) $(MAIN_SRCS) -o $(BUILD_DIR)/salr_dft -lm

# ── Tests ─────────────────────────────────────────────────────────────────────
tests: test_solver test_potential

test_solver: $(TESTS)/test_solver.c $(SRC_CPU)/solver_cpu.c $(SRC_CPU)/math_utils_cpu.c
	$(CC) $(CFLAGS) -I$(INCLUDE) \
		$(TESTS)/test_solver.c \
		$(SRC_CPU)/solver_cpu.c \
		$(SRC_CPU)/math_utils_cpu.c \
		-o $(BUILD_DIR)/test_solver -lm

test_potential: $(TESTS)/test_potential.c $(SRC_CPU)/potential_cpu.c
	$(CC) $(CFLAGS) -I$(INCLUDE) \
		$(TESTS)/test_potential.c \
		$(SRC_CPU)/potential_cpu.c \
		-o $(BUILD_DIR)/test_potential -lm

# ═══════════════════════════════════════════════════════════════════════════════
#  CLEAN
# ═══════════════════════════════════════════════════════════════════════════════
clean:
	rm -rf $(BUILD_DIR)/*
