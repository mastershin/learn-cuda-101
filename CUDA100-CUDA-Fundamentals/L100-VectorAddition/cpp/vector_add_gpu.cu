#include <cuda_runtime.h>
#include <chrono>
#include <iostream>

#define SIZE 1000 * 1000 * 200
#define LOOP 20

#define BLOCK_SIZE 256

using namespace std;
// Utility function to get the current time in seconds
using time_point = chrono::time_point<chrono::high_resolution_clock>;

time_point now() {
  return chrono::high_resolution_clock::now();
}
auto time_diff(time_point a, time_point b) {
  // Return Type: std::chrono::duration<double>
  return chrono::duration_cast<std::chrono::duration<double>>(b - a);
}

// GPU kernel for vector addition
// __global__ void gpuVectorAdd(int *x, int *y, int *out, int size)
// {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < size)
//     {
//         out[idx] = x[idx] + y[idx];
//     }
// }
__global__ void gpuVectorAdd(const float* x, const float* y, float* out,
                             int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    out[idx] = x[idx] + y[idx];
  }
}

// Function to perform vector addition on the GPU
template <typename T>
class VectorAdditionCUDA {
 private:
  T* d_x = nullptr;
  T* d_y = nullptr;
  T* d_out = nullptr;
  int size;  // Store the vector size

 public:
  VectorAdditionCUDA(int size) : size(size) {
    // Allocate device memory with error checking
    cudaMalloc((void**)&d_x, size * sizeof(T));
    cudaCheckError("cudaMalloc d_x");
    cudaMalloc((void**)&d_y, size * sizeof(T));
    cudaCheckError("cudaMalloc d_y");
    cudaMalloc((void**)&d_out, size * sizeof(T));
    cudaCheckError("cudaMalloc d_out");
  }

  ~VectorAdditionCUDA() {
    // Free device memory with error checking
    if (d_x) {
      cudaFree(d_x);
      cudaCheckError("cudaFree d_x");
    }
    if (d_y) {
      cudaFree(d_y);
      cudaCheckError("cudaFree d_y");
    }
    if (d_out) {
      cudaFree(d_out);
      cudaCheckError("cudaFree d_out");
    }
  }

  void run_kernel(T a, T* x, T* y, T* out) {
    // Copy vectors x and y from host to device
    cudaMemcpy(d_x, x, size * sizeof(T), cudaMemcpyHostToDevice);
    cudaCheckError("cudaMemcpy x");
    cudaMemcpy(d_y, y, size * sizeof(T), cudaMemcpyHostToDevice);
    cudaCheckError("cudaMemcpy y");

    // Calculate the number of blocks needed
    int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Launch the GPU kernel for vector addition with error checking
    gpuVectorAdd<<<numBlocks, BLOCK_SIZE>>>(a, d_x, d_y, d_out, size);
    cudaCheckError("kernel launch");

    // Wait for GPU to finish execution
    cudaDeviceSynchronize();

    // Copy vector c from device to host
    cudaMemcpy(out, d_out, size * sizeof(T), cudaMemcpyDeviceToHost);
    cudaCheckError("cudaMemcpy out");
  }

  // Helper function for CUDA error checking
  inline void cudaCheckError(const char* msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cerr << "CUDA error: " << msg << " (" << cudaGetErrorString(err)
                << ")" << std::endl;
      exit(EXIT_FAILURE);
    }
  }
};

void initialize_data(float* x, float* y, int size) {
  for (int i = 0; i < size; i++) {
    x[i] = i;
    y[i] = i * 2;
  }
}

int main() {
  // Allocate memory for arrays a, b, and c on the host
  float* x = new float[SIZE];
  float* y = new float[SIZE];
  float* out = new float[SIZE];

  // Initialize vectors a and b
  initialize_data(x, y, SIZE);

  // Call the function to perform vector addition on the GPU

  // Start GPU timer
  auto start_gpu = now();

  auto vector_add_cuda = VectorAdditionCUDA<float>(SIZE);

  for (int i = 0; i < LOOP; i++) {
    cout << "." << flush;
    vector_add_cuda.run_kernel(x, y, out);
  }

  // Stop GPU timer
  auto end_gpu = now();
  auto gpu_duration = time_diff(start_gpu, end_gpu);

  // Print GPU execution time
  cout << endl;
  cout << "GPU time: " << gpu_duration.count() << " seconds" << endl;

  // Calculate the avg using a loop
  float avg = 0.0f;
  for (int i = 0; i < SIZE; ++i) {
    avg += out[i];
  }
  avg /= SIZE;
  cout << "Avg: " << avg << endl;

  // Clean up memory on the host
  delete[] x;
  delete[] y;
  delete[] out;

  return 0;
}
