#!/usr/bin/env bash
# Reference: https://www.intel.com/content/www/us/en/developer/tools/isa-extensions/overview.html

# Check for g++ compiler
if ! command -v g++ >/dev/null 2>&1; then
  echo "g++ not found. Please install it using your package manager."
  echo "For example, on Ubuntu/Debian:"
  echo "  sudo apt update"
  echo "  sudo apt install build-essential"
  exit 1
fi

echo "g++ found. Compiling CPU code..."
g++ --std=c++17 vector_add_opencl.cpp -framework OpenCL -o vector_add_opencl.ex
[ $? -eq 0 ] || { echo "Compilation failed!"; exit 1; }

echo "Running OpenCL executables..."
./vector_add_opencl.ex
