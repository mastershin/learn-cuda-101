# Vector Addition using CUDA, CPU, CPU/multi-core and CPU/AVX
Demonstrates the addition of two 1D float arrays in C/C++ and Python.
The code measures the speed at which two 1D vectors can be added.

## Performance (100 loops, 200 million elements)
Note: GPU performance might have been slow due to CUDA memory to host's memory movement.  GPU kernel operation may be too simple.
TODO: Optimize further.

| Device | CPU | CPU/multi-core | CPU/AVX | CUDA (1080ti) | CUDA (RTX8000) | CUDA (A100) |
| - | - | - | - | - | - | - |
| Apple M1 | 26.3s | 4.5s (10 threads) | - | - | - | - |
| Intel(R) Xeon(R) CPU E5-2687W v4 @ 3.00GHz | 62.1s | 6.6s (24 threads) | 26.2s (AVX2) | 24s | - | - |
| Intel(R) Xeon(R) Gold 6230 CPU @ 2.10GHz | 50.7s | 4s (80 threads) | 23.8s (AVX512f) | - | - | 55s |
| Intel(R) Xeon(R) Gold 6240 CPU @ 2.60GHz | 49s | 3s (72 threads) | 21.6s (AVX512f) | - | 48.4s | - |