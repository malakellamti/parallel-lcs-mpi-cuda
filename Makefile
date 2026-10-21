
CC = gcc
MPICC = mpicc
NVCC = nvcc

CFLAGS = -Wall -Wextra -Wpedantic -O2
CUDAFLAGS = -O2

.PHONY: all sequential test clean

all: sequential

sequential: lcs_sequential

lcs_sequential: src/lcs_sequential.c
	$(CC) $(CFLAGS) $< -o $@

test: sequential
	bash tests/test_sequential.sh

clean:
	rm -f lcs_sequential lcs_mpi lcs_cuda


