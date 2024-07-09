#!/usr/bin/env bash
# CUDA: https://developer.nvidia.com/cuda-downloads
# Reference: https://github.com/nvidia/cuda-samples

# Check for nvcc compiler
if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc not found. Please install the CUDA Toolkit from NVIDIA."
  echo "Visit: https://developer.nvidia.com/cuda-downloads"
  exit 1
fi

echo
echo "nvcc found. Compiling GPU code..."
nvcc matmul_gpu.cu -o matmul_gpu.ex
[ $? -eq 0 ] || { echo "Compilation failed!"; exit 1; }

echo "Running GPU executables..."
./matmul_gpu.ex $@
