
CC = gcc
MPICC = mpicc
NVCC = nvcc

CFLAGS = -Wall -Wextra -Wpedantic -O2
CUDAFLAGS = -O2

.PHONY: all sequential mpi test test-sequential test-mpi clean

all: sequential mpi

sequential: lcs_sequential

mpi: lcs_mpi

lcs_sequential: src/lcs_sequential.c
	$(CC) $(CFLAGS) $< -o $@

lcs_mpi: src/lcs_mpi.c
	$(MPICC) $(CFLAGS) $< -o $@

test: test-sequential test-mpi

test-sequential: lcs_sequential
	bash tests/test_sequential.sh

test-mpi: lcs_mpi
	bash tests/test_mpi.sh

clean:
	rm -f lcs_sequential lcs_mpi lcs_cuda
