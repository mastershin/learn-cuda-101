# Vector Addition using CUDA, CPU, CPU/multi-core and CPU/AVX
Demonstrates the addition of two 1D float arrays in C/C++ and Python.
The code measures the speed at which two 1D vectors can be added.

## Performance (100 loops, 200 million elements)
| Device | CPU | CPU/multi-core | CPU/AVX2 | CUDA (1080ti) | CUDA (RTX8000) | CUDA (A100) |
| - | - | - | - | - | - | - |
| Apple M1 | 26.3 sec | 4.5 sec (10 threads) | - | - | - | - |