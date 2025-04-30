# Makefile to compile and run all neural network versions
# Usage: 
#   make run_v1    (CPU version with -O2 optimization)
#   make run_v2    (GPU version with -O2 optimization)
#   make run_v3    (GPU version with -O2 optimization)
#   make run_v4    (GPU version with -O2 optimization)
#   make clean     (Remove binaries)

# --- Compiler Settings ---
# C (CPU) Version
CC      = gcc
CFLAGS  = -Wall -O2  

# CUDA (GPU) Versions
NVCC    = nvcc
NVFLAGS = -lcublas 

# --- Targets ---
# v1 (CPU)
v1_nn: v1_nn.c
	$(CC) $(CFLAGS) $< -o $@ -lm

run_v1: v1_nn
	./v1_nn

# v2 (GPU)
v2_nn: v2_nn.cu
	$(NVCC) $(NVFLAGS) $< -o $@

run_v2: v2_nn
	./v2_nn

# v3 (GPU)
v3_nn: v3_nn.cu
	$(NVCC) $(NVFLAGS) $< -o $@

run_v3: v3_nn
	./v3_nn

# v4 (GPU)
v4_nn: v4_nn.cu
	$(NVCC) $(NVFLAGS) $< -o $@

run_v4: v4_nn
	./v4_nn

# Cleanup
clean:
	rm -f v1_nn v2_nn v3_nn v4_nn

.PHONY: run_v1 run_v2 run_v3 run_v4 clean