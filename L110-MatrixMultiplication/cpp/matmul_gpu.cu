#include <chrono>
#include <cstdlib>  // For atoi
#include <iostream>
#include <tuple>

#include <cuda_runtime.h>

#define SIZE 1000 * 1000 * 10
#define BLOCK_SIZE 256
#define LOOP 200

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

// Utility function to get the current time in seconds
using time_point = std::chrono::time_point<std::chrono::high_resolution_clock>;

time_point now() {
  return std::chrono::high_resolution_clock::now();
}
auto time_diff(time_point a, time_point b) {
  // Return Type: std::chrono::duration<double>
  return std::chrono::duration_cast<std::chrono::duration<double>>(b - a);
}

// GPU kernel for vector addition
__global__ void gpuVectorAdd(float a, float* x, float* y, float* out,
                             int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    out[idx] = a * x[idx] + y[idx];
  }
}

// Function to perform vector addition on the GPU
void performVectorAddition(float a, float* x, float* y, float* out, int size) {
  // Allocate memory for arrays d_a, d_b, and d_c on the device
  float *d_x, *d_y, *d_out;
  gpuErrorCheck(cudaMalloc((void**)&d_x, size * sizeof(float)));
  gpuErrorCheck(cudaMalloc((void**)&d_y, size * sizeof(float)));
  gpuErrorCheck(cudaMalloc((void**)&d_out, size * sizeof(float)));

  // Copy vectors a and b from host to device
  gpuErrorCheck(
      cudaMemcpy(d_x, x, size * sizeof(float), cudaMemcpyHostToDevice));
  gpuErrorCheck(
      cudaMemcpy(d_x, y, size * sizeof(float), cudaMemcpyHostToDevice));

  // Calculate the number of blocks needed
  int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

  // Launch the GPU kernel for vector addition
  gpuVectorAdd<<<numBlocks, BLOCK_SIZE>>>(a, d_x, d_y, d_out, size);

  // Wait for GPU to finish execution
  cudaDeviceSynchronize();

  // Copy vector c from device to host
  gpuErrorCheck(
      cudaMemcpy(out, d_out, size * sizeof(float), cudaMemcpyDeviceToHost));

  // Free memory on the device
  gpuErrorCheck(cudaFree(d_x));
  gpuErrorCheck(cudaFree(d_y));
  gpuErrorCheck(cudaFree(d_out));
}

void initialize_data(float* x, float* y, int size) {
  for (int i = 0; i < size; i++) {
    x[i] = i;
    y[i] = i * 2;
  }
}

void get_small_matrix_size(int& m, int& n, int& k) {
  // (200x150, 150x100 --> 200x100)
  // m=200, n=150, k=100 --> sum --> -200 (~1.3 sec on CPU)
  m = 200;
  n = 150;
  k = 100;
}

void get_medium_matrix_size(int& m, int& n, int& k) {
  // 500x300, 300x200, 500x200 --> sum --> -400 (~13 sec on CPU)
  m = 500;
  n = 300;
  k = 200;
}

void get_large_matrix_size(int& m, int& n, int& k) {
  // More realistic small-scale LLM size
  // 4096x1024, 1024x1024, 4096x1024 --> sum --> ?
  // int m = 4096, n = 1024, k = 1024;

  m = 4096;
  n = 1024;
  k = 1024;
}

std::tuple<int, int, int> parse_command_args(int argc, char* argv[]) {
  int m, n, k;

  if (argc == 2) {
    std::string size_arg = argv[1];
    if (size_arg == "s") {
      get_small_matrix_size(m, n, k);
    } else if (size_arg == "m") {
      get_medium_matrix_size(m, n, k);
    } else if (size_arg == "l") {
      get_large_matrix_size(m, n, k);
    } else {
      std::cerr
          << "Invalid size argument. Use 's', 'm', 'l' or specify dimensions."
          << std::endl;
      exit(1);
    }
  } else if (argc == 4) {
    m = std::atoi(argv[1]);
    n = std::atoi(argv[2]);
    k = std::atoi(argv[3]);
  } else {
    std::cerr << "Invalid arguments. Use 's', 'm', 'l' for predefined sizes or "
                 "specify dimensions m, n, k."
              << std::endl;
    exit(1);
  }

  return std::make_tuple(m, n, k);
}

int main(int argc, char* argv[]) {

  auto [m, n, k] = parse_command_args(argc, argv);

  std::cout << "Matrix Multiplication: A(" << m << "x" << k << ") * B(" << k
            << "x" << n << ") = C(" << m << "x" << n << ")" << std::endl;

  // Allocate memory for arrays a, b, and c on the host
  float a = 2.0;

  //   float* x = new float[SIZE];
  //   float* y = new float[SIZE];
  //   float* out = new float[SIZE];

  float *x, *y, *out;
  gpuErrorCheck(cudaMallocHost(&x, SIZE * sizeof(float)));
  gpuErrorCheck(cudaMallocHost(&y, SIZE * sizeof(float)));
  gpuErrorCheck(cudaMallocHost(&out, SIZE * sizeof(float)));

  // Initialize vectors a and b
  initialize_data(x, y, SIZE);

  // Call the function to perform vector addition on the GPU
  // Start GPU timer
  auto start_gpu = now();

  for (int i = 0; i < LOOP; i++) {
    std::cout << "." << std::flush;
    performVectorAddition(a, x, y, out, SIZE);
  }

  // Stop GPU timer
  auto end_gpu = now();
  auto gpu_duration = time_diff(start_gpu, end_gpu);

  // Print GPU execution time
  std::cout << std::endl;
  std::cout << "GPU time: " << gpu_duration.count() << " seconds" << std::endl;

  // Clean up memory on the host
  //delete[] x;
  //delete[] y;
  //delete[] out;

  cudaFreeHost(x);
  cudaFreeHost(y);
  cudaFreeHost(out);

  return 0;
}
