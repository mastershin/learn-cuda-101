#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>

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
__global__ void sum_kernel(float* d_data, float* d_result, int N) {
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

__global__ void min_kernel(float* d_data, float* d_result, int N) {
  // Calculate the thread ID and block ID
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int i = tid;
  // Perform reduction within a thread block
  if (tid < N) {
    atomicMin((int*)d_result, __float_as_int(d_data[i]));
    while (i < N) {
      atomicMin((int*)d_result, __float_as_int(d_data[i]));
      i += blockDim.x * gridDim.x;
    }
  }
}

__global__ void max_kernel(float* d_data, float* d_result, int N) {
  // Calculate the thread ID and block ID
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int i = tid;
  // Perform reduction within a thread block
  if (tid < N) {
    atomicMax((int*)d_result, __float_as_int(d_data[i]));
    while (i < N) {
      atomicMax((int*)d_result, __float_as_int(d_data[i]));
      i += blockDim.x * gridDim.x;
    }
  }
}

__global__ void avg_kernel(float* d_data, float* d_result, int N) {
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

// GPU Kernel for Bitonic Sort
__global__ void bitonic_sort_step(float* d_data, int j, int k) {
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int ixj = tid ^ j;

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

void bitonic_sort(float* d_data, int N) {
  for (int k = 2; k <= N; k <<= 1) {
    for (int j = k >> 1; j > 0; j = j >> 1) {
      bitonic_sort_step<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
          d_data, j, k);
      cudaDeviceSynchronize();
    }
  }
}

__global__ void median_kernel(float* d_data, float* d_result, int N) {
  // Calculate the thread ID and block ID
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  // Calculate the median based on the sorted data
  if (tid == N / 2) {
    *d_result = d_data[tid];
  }
}

int parse_size(const std::string& size_str) {
  char suffix = size_str.back();
  int base_size = std::stoi(size_str.substr(0, size_str.size() - 1));
  if (suffix == 'M' || suffix == 'm') {
    return base_size * 1024 * 1024;
  } else if (suffix == 'B' || suffix == 'b') {
    return base_size * 1024 * 1024 * 1024;
  } else {
    throw std::invalid_argument(
        "Invalid size suffix. Use 'M' for million or 'B' for billion.");
  }
}

int main(int argc, char* argv[]) {
  // Check for command line arguments
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " [size] [operation]" << std::endl;
    std::cerr << "Supported operations: sum, min, max, avg, median, sort"
              << std::endl;
    return 1;
  }

  std::string size_str = argv[1];
  std::string operation = argv[2];

  int N;
  try {
    N = parse_size(size_str);
  } catch (const std::invalid_argument& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }

  // Initialize the random number generator
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-5.0, 5.0);

  // Allocate memory on the host using cudaMallocHost
  float* h_data;
  gpuErrorCheck(cudaMallocHost(&h_data, N * sizeof(float)));

  float h_result = 0.0f;

  // Allocate memory on the device
  float *d_data, *d_result;
  gpuErrorCheck(cudaMalloc(&d_data, N * sizeof(float)));
  gpuErrorCheck(cudaMalloc(&d_result, sizeof(float)));

  // Initialize the array with random numbers
  for (int i = 0; i < N; i++) {
    h_data[i] = dis(gen);
  }

  // Copy data from host to device
  gpuErrorCheck(
      cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice));

  // Start the timer
  auto start = std::chrono::high_resolution_clock::now();

  // Launch the appropriate GPU kernel based on the operation type
  if (operation == "sum") {
    sum_kernel<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_data,
                                                                  d_result, N);
  } else if (operation == "min") {
    float init_val = std::numeric_limits<float>::max();
    gpuErrorCheck(
        cudaMemcpy(d_result, &init_val, sizeof(float), cudaMemcpyHostToDevice));
    min_kernel<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_data,
                                                                  d_result, N);
  } else if (operation == "max") {
    float init_val = std::numeric_limits<float>::min();
    gpuErrorCheck(
        cudaMemcpy(d_result, &init_val, sizeof(float), cudaMemcpyHostToDevice));
    max_kernel<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_data,
                                                                  d_result, N);
  } else if (operation == "avg") {
    avg_kernel<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_data,
                                                                  d_result, N);
  } else if (operation == "median") {
    bitonic_sort(d_data, N);
    median_kernel<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        d_data, d_result, N);
  } else if (operation == "sort") {
    bitonic_sort(d_data, N);
    gpuErrorCheck(
        cudaMemcpy(h_data, d_data, N * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "First 5 elements: ";
    for (int i = 0; i < 5; i++) {
      std::cout << h_data[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Last 5 elements: ";
    for (int i = N - 5; i < N; i++) {
      std::cout << h_data[i] << " ";
    }
    std::cout << std::endl;
  } else {
    std::cerr << "Invalid operation type." << std::endl;
    return 1;
  }

  // Synchronize the GPU
  gpuErrorCheck(cudaDeviceSynchronize());

  // Stop the timer
  auto end = std::chrono::high_resolution_clock::now();

  if (operation != "sort") {
    // Copy the result from device to host
    gpuErrorCheck(
        cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost));

    // Print the results
    std::cout << "Operation: " << operation << std::endl;
    std::cout << "GPU result: " << std::fixed << std::setprecision(6)
              << h_result << std::endl;
  }

  std::cout << "Execution time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                     start)
                   .count()
            << " ms" << std::endl;

  // Free the memory
  gpuErrorCheck(cudaFree(d_data));
  gpuErrorCheck(cudaFree(d_result));
  gpuErrorCheck(cudaFreeHost(h_data));

  return 0;
}
