#!/usr/bin/env bash

./matmul_cpu/run-cpu.sh $@

./matmul_avx/run-cpu-avx.sh $@

./matmul_cuda/run-gpu-cuda.sh $@
