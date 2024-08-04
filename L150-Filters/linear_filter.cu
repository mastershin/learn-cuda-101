
#include "linear_filter.hpp"

__device__ __constant__ int d_rows;
__device__ __constant__ int d_columns;

__global__ void __cuda_kernel_applySimpleLinearBlurFilter(uchar* r, uchar* g,
                                                   uchar* b) {
  extern __shared__ uchar sharedMem[];

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  int sharedWidth = blockDim.x + 2;  // Include left and right halo
  int sharedIndex =
      (threadIdx.y * sharedWidth + threadIdx.x + 1) * 3;  // +1 for left halo

  int globalIdx = y * d_columns + x;
  // Load data into shared memory
  if (x < d_columns && y < d_rows) {
    sharedMem[sharedIndex + 0] = r[globalIdx];
    sharedMem[sharedIndex + 1] = g[globalIdx];
    sharedMem[sharedIndex + 2] = b[globalIdx];

    // Load halo elements
    if (threadIdx.x == 0 && x > 0) {
      sharedMem[sharedIndex - 3] = r[globalIdx - 1];
      sharedMem[sharedIndex - 2] = g[globalIdx - 1];
      sharedMem[sharedIndex - 1] = b[globalIdx - 1];
    }
    if (threadIdx.x == blockDim.x - 1 && x < d_columns - 1) {
      sharedMem[sharedIndex + 3] = r[globalIdx + 1];
      sharedMem[sharedIndex + 4] = g[globalIdx + 1];
      sharedMem[sharedIndex + 5] = b[globalIdx + 1];
    }
  }

  __syncthreads();

  // Apply the blur filter (ignore boundary pixels for simplicity)
  if (x > 0 && x < d_columns - 1 && y < d_rows) {
    uchar newR = (sharedMem[sharedIndex - 3] + sharedMem[sharedIndex + 0] +
                  sharedMem[sharedIndex + 3]) /
                 3;
    uchar newG = (sharedMem[sharedIndex - 2] + sharedMem[sharedIndex + 1] +
                  sharedMem[sharedIndex + 4]) /
                 3;
    uchar newB = (sharedMem[sharedIndex - 1] + sharedMem[sharedIndex + 2] +
                  sharedMem[sharedIndex + 5]) /
                 3;

    // Write the new values back to global memory
    r[globalIdx] = newR;
    g[globalIdx] = newG;
    b[globalIdx] = newB;
  }
}

__host__ void allocateDeviceMemory(int rows, int columns) {

  //Allocate device constant symbols for rows and columns
  cudaMemcpyToSymbol(d_rows, &rows, sizeof(int), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(d_columns, &columns, sizeof(int), 0,
                     cudaMemcpyHostToDevice);
}

__host__ void kernel_stub_applySimpleLinearBlurFilter(uchar* r, uchar* g, uchar* b, int rows, int columns,
                            int threadsPerBlock) {

  dim3 gridSize((columns + 16 - 1) / 16, (rows + 16 - 1) / 16);
  dim3 blockSize(16, 16);
  int sharedMemSize = (gridSize.x + 2) * gridSize.y * 3 * sizeof(uchar);
  __cuda_kernel_applySimpleLinearBlurFilter<<<gridSize, blockSize, sharedMemSize>>>(
      r, g, b);

}