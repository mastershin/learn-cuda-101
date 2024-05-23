#!/usr/bin/env bash
# Reference: https://github.com/nvidia/cuda-samples

# Check for g++ compiler
if ! command -v g++ >/dev/null 2>&1; then
  echo "g++ not found. Please install it using your package manager."
  echo "For example, on Ubuntu/Debian:"
  echo "  sudo apt update"
  echo "  sudo apt install build-essential"
  exit 1
fi

# Check for nvcc compiler
if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc not found. Please install the CUDA Toolkit from NVIDIA."
  echo "Visit: https://developer.nvidia.com/cuda-downloads"
  exit 1
fi

echo "g++ found. Compiling CPU code..."
g++ vector_add_cpu.cpp -o vector_add_cpu.ex

echo "nvcc found. Compiling GPU code..."
nvcc vector_add_gpu.cu -o vector_add_gpu.ex

# Run the CPU and GPU executables
echo "Running CPU executables..."
./vector_add_cpu.ex

echo "Running GPU executables..."
./vector_add_gpu.ex
