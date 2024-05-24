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

echo "g++ found. Compiling CPU code..."
g++ matrix_multiplication_cpu.cpp -o matrix_multiplication_cpu.ex

# Run the CPU and GPU executables
echo "Running CPU executables..."
./matrix_multiplication_cpu.ex


# Function to compile with specific instruction set
echo
avx_compile_with() {
  
  echo "Compiling ..."
  # g++ -march=native -mavx -mavx2 -mavx512 matrix_multiplication_cpu_avx.cpp -o matrix_multiplication_cpu_avx.ex
  g++ --std c++21 -march=native -mavx -mavx2 matrix_multiplication_cpu_avx.cpp -o matrix_multiplication_cpu_avx.ex

  # Run the CPU and GPU executables
  echo "Running CPU (AVX=$1) executables..."
  ./matrix_multiplication_cpu_avx.ex
}

# Check for CPU instruction set support
if lscpu | grep -q "avx"; then
  # Although we compile with highest supported AVX instruction set
  # at runtime, the code will run on any CPU that supports AVX
  avx_compile_with "avx512f"

#if lscpu | grep -q "avx512"; then
#  avx_compile_with "avx512f"
#elif lscpu | grep -q "avx2"; then
#  avx_compile_with "avx2"
#elif lscpu | grep -q "avx"; then
#  avx_compile_with "avx"
else
  echo "AVX-512 support not detected. Skipping AVX-512 compilation."
fi

########################################################################
# Check for nvcc compiler
if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc not found. Please install the CUDA Toolkit from NVIDIA."
  echo "Visit: https://developer.nvidia.com/cuda-downloads"
  exit 1
fi


echo
echo "nvcc found. Compiling GPU code..."
nvcc matrix_multiplication_gpu.cu -o matrix_multiplication_gpu.ex



echo "Running GPU executables..."
./matrix_multiplication_gpu.ex
