#!/usr/bin/env bash

# Check for g++ compiler
if ! command -v g++ >/dev/null 2>&1; then
  echo "g++ not found. Please install it using your package manager."
  echo "For example, on Ubuntu/Debian:"
  echo "  sudo apt update"
  echo "  sudo apt install build-essential"
  exit 1
fi

# Function to compile with specific instruction set
echo
echo "Compiling with ..."
if lscpu | grep -q "avx512"; then
  echo "** avx512f"
elif lscpu | grep -q "avx2"; then
  echo "** avx2"
elif lscpu | grep -q "avx"; then
  echo "** avx"
else
  echo "** AVX support not detected."
fi

# fma: fused multiply-add
g++ --std=c++17 -mavx -mavx2 -mavx512f -mfma matmul_cpu_avx.cpp avx.cpp main.cpp -o matmul_cpu_avx.ex

# Run the CPU and GPU executables
echo "Running CPU (AVX=$1) executables..."
./matmul_cpu_avx.ex $@
