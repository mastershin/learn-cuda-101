#!/usr/bin/env bash

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
  echo "AVX support not detected."
fi

g++ --std=c++17 -mavx512f vector_add_cpu_avx.cpp -o vector_add_cpu_avx.ex
[ $? -eq 0 ] || { echo "Compilation failed!"; exit 1; }

# Run the CPU and GPU executables
echo "Running CPU/AVX executables..."

./vector_add_cpu_avx.ex
