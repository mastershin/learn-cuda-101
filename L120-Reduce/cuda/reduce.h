#pragma once

#include <cuda_runtime.h>

// Error checking macro
#define gpuErrorCheck(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

void sum_kernel(float* d_data, float* d_result, int N);
void min_kernel(float* d_data, float* d_result, int N);
void max_kernel(float* d_data, float* d_result, int N);
void avg_kernel(float* d_data, float* d_result, int N);
// void median_kernel(float* d_data, float* d_result, int N);

void bitonic_sort(float* d_data, int N);
