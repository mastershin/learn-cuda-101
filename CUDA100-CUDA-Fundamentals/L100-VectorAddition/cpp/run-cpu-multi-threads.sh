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

echo "g++ found. Compiling CPU/Multi-threads code..."
g++ --std=c++20 vector_add_cpu_threads.cpp -o vector_add_cpu_threads.ex
[ $? -eq 0 ] || { echo "Compilation failed!"; exit 1; }

# Run the CPU and GPU executables
echo "Running CPU/Multi-threads executables..."
./vector_add_cpu_threads.ex
