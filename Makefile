
CC = gcc
MPICC = mpicc
NVCC = nvcc
CUDA_HOST_COMPILER = g++-14

CFLAGS = -Wall -Wextra -Wpedantic -O2
CUDAFLAGS = -O2

.PHONY: all sequential mpi cuda test test-sequential test-mpi test-cuda clean

all: sequential mpi

sequential: lcs_sequential

mpi: lcs_mpi

cuda: lcs_cuda

lcs_sequential: src/lcs_sequential.c
	$(CC) $(CFLAGS) $< -o $@

lcs_mpi: src/lcs_mpi.c
	$(MPICC) $(CFLAGS) $< -o $@

lcs_cuda: src/lcs_cuda.cu
	$(NVCC) -ccbin $(CUDA_HOST_COMPILER) $(CUDAFLAGS) $< -o $@

test: test-sequential test-mpi test-cuda

test-sequential: lcs_sequential
	bash tests/test_sequential.sh

test-mpi: lcs_mpi
	bash tests/test_mpi.sh

test-cuda: lcs_cuda
	bash tests/test_cuda.sh

clean:
	rm -f lcs_sequential lcs_mpi lcs_cuda
