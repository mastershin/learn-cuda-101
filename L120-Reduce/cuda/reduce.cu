#include <cuda_runtime.h>
#include <float.h>
//#include <algorithm>
//#include <chrono>
//#include <cmath>
//#include <iomanip>
#include <iostream>
//#include <random>
//#include <string>

// Define the number of threads per block
#define BLOCK_SIZE 256

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

// Define the GPU kernel functions for different reductions
__global__ void cuda_sum_kernel(float* d_data, float* d_result, int N) {
  // Calculate the thread ID and block ID
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int i = tid;
  // Perform reduction within a thread block
  if (tid < N) {
    while (i < N) {
      atomicAdd(d_result, d_data[i]);
      i += blockDim.x * gridDim.x;
    }
  }
}

__global__ void cuda_min_kernel(float* d_data, float* d_result, int N) {
  // Calculate thread ID and block ID
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int i = tid;

  // Shared memory for reduction within the block
  __shared__ float block_min;

  // Load initial value into shared memory
  if (tid < N) {
    block_min = d_data[i];
  } else {
    block_min = FLT_MAX;  // Initialize with a large value
  }
  __syncthreads();

  // Perform reduction within the thread block
  while (i < N) {
    block_min = min(block_min, d_data[i]);
    i += blockDim.x * gridDim.x;
  }
  __syncthreads();

  // Thread 0 writes the block minimum to global memory
  if (threadIdx.x == 0) {
    d_result[blockIdx.x] = block_min;
  }
}
// Function to launch the kernel and perform global reduction
void min_kernel(float* d_data, float* d_result, int N) {
  // Launch the kernel
  cuda_min_kernel<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
      d_data, d_result, N);

  // Perform global reduction (using a single block with all threads)
  cuda_min_kernel<<<1, BLOCK_SIZE>>>(d_result, d_result,
                                     (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

  // Final result is stored in d_result[0]
}

__global__ void cuda_max_kernel(float* d_data, float* d_result, int N) {
  // Calculate thread ID and block ID
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int i = tid;

  // Shared memory for reduction within the block
  __shared__ float block_max;

  // Load initial value into shared memory
  if (tid < N) {
    block_max = d_data[i];
  } else {
    block_max = -FLT_MAX;  // Initialize with a small value
  }
  __syncthreads();

  // Perform reduction within the thread block
  while (i < N) {
    block_max = max(block_max, d_data[i]);
    i += blockDim.x * gridDim.x;
  }
  __syncthreads();

  // Thread 0 writes the block maximum to global memory
  if (threadIdx.x == 0) {
    d_result[blockIdx.x] = block_max;
  }
}
// Function to launch the kernel and perform global reduction
void max_kernel(float* d_data, float* d_result, int N) {
  // Launch the kernel
  cuda_max_kernel<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
      d_data, d_result, N);

  // Perform global reduction (using a single block with all threads)
  cuda_max_kernel<<<1, BLOCK_SIZE>>>(d_result, d_result,
                                     (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

  // Final result is stored in d_result[0]
}

__global__ void cuda_avg_kernel(float* d_data, float* d_result, int N) {
  // Calculate the thread ID and block ID
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int i = tid;
  // Perform reduction within a thread block
  if (tid < N) {
    atomicAdd(d_result, d_data[i]);
    while (i < N) {
      atomicAdd(d_result, d_data[i]);
      i += blockDim.x * gridDim.x;
    }
  }
  // Calculate average after reduction
  __syncthreads();
  if (tid == 0) {
    *d_result /= N;
  }
}

// Median Kernel doesn't need to be run in parallel
// if we sort the data first.
// __global__ void cuda_median_kernel(float* d_data, float* d_result, int N) {
//   // Calculate the thread ID and block ID
//   int tid = threadIdx.x + blockIdx.x * blockDim.x;

//   // Calculate the median based on the sorted data
//   if (tid == N / 2) {
//     *d_result = d_data[tid];
//   }
// }

void sum_kernel(float* d_data, float* d_result, int N) {
  cuda_sum_kernel<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
      d_data, d_result, N);
}

void avg_kernel(float* d_data, float* d_result, int N) {
  cuda_avg_kernel<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
      d_data, d_result, N);
}
// void median_kernel(float* d_data, float* d_result, int N) {
//   cuda_median_kernel<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
//       d_data, d_result, N);
// }

// GPU Kernel for Bitonic Sort
__global__ void cuda_bitonic_sort_step(float* d_data, int j, int k, int N) {
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int ixj = tid ^ j;

  // Ensure both tid and ixj are within bounds
  //if (tid < N && ixj < N) 
  {
    /* The threads with the lowest ids sort the array. */
    if (ixj > tid) {
      if ((tid & k) == 0) {
        /* Sort ascending */
        if (d_data[tid] > d_data[ixj]) {
          // Swap
          float temp = d_data[tid];
          d_data[tid] = d_data[ixj];
          d_data[ixj] = temp;
        }
      } else {
        /* Sort descending */
        if (d_data[tid] < d_data[ixj]) {
          // Swap
          float temp = d_data[tid];
          d_data[tid] = d_data[ixj];
          d_data[ixj] = temp;
        }
      }
    }
  }
}
void bitonic_sort(float* d_data, int N) {
  for (int k = 2; k <= N; k <<= 1) {
    for (int j = k >> 1; j > 0; j = j >> 1) {
      cuda_bitonic_sort_step<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
          d_data, j, k, N);
      cudaDeviceSynchronize();
    }
  }
}
