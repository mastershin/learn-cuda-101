#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <thread>

#include <cuda_runtime.h>
#include "reduce.h"

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

// Function to fill a portion of the array with random values in parallel
void fill_random(float* start, float* end) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(-5.0, 5.0);
  std::generate(start, end, [&]() { return dis(gen); });
}

void fill_random_for_loop(float* start, float* end) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(-5.0, 5.0);

  // traditional way of filling the array
  for (float* i = start; i < end; i++) {
    *i = dis(gen);
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
  std::cout << "*** " << operation << " ***" << std::endl;

  int N;
  try {
    N = parse_size(size_str);
  } catch (const std::invalid_argument& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }

  // Allocate memory on the host using cudaMallocHost
  std::cout << "Allocating memory on the host..." << std::endl;
  float* h_data;
  gpuErrorCheck(cudaMallocHost(&h_data, N * sizeof(float)));

  float h_result = 0.0f;

  // Allocate memory on the device
  std::cout << "Allocating memory on the device..." << std::endl;
  float *d_data, *d_result;
  gpuErrorCheck(cudaMalloc(&d_data, N * sizeof(float)));
  gpuErrorCheck(cudaMalloc(&d_result, N * sizeof(float)));

  // Initialize the array with random numbers
  std::cout << "Initializing the array with random numbers..." << std::endl;

  // Create threads
  int num_threads = std::thread::hardware_concurrency();
  std::vector<std::thread> threads;
  int chunk_size = N / num_threads;
  for (int i = 0; i < num_threads; ++i) {
    float* start = h_data + i * chunk_size;
    float* end = (i == num_threads - 1) ? h_data + N : start + chunk_size;
    threads.emplace_back(fill_random, start, end);
  }

  // Join threads
  for (auto& t : threads) {
    t.join();
  }
  // Print some values to verify
  for (int i = 0; i < 10; ++i) {
    std::cout << h_data[i] << " ";
  }
  std::cout << std::endl;

  // Copy data from host to device
  std::cout << "Copying data from host to device..." << std::endl;
  gpuErrorCheck(
      cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice));

  // Start the timer
  std::cout << "Running the " << operation << " ..." << std::endl;
  auto start = std::chrono::high_resolution_clock::now();

  // Launch the appropriate GPU kernel based on the operation type
  if (operation == "sum") {
    sum_kernel(d_data, d_result, N);
  } else if (operation == "min") {
    float init_val = std::numeric_limits<float>::max();
    gpuErrorCheck(
        cudaMemcpy(d_result, &init_val, sizeof(float), cudaMemcpyHostToDevice));
    min_kernel(d_data, d_result, N);
  } else if (operation == "max") {
    float init_val = std::numeric_limits<float>::min();
    gpuErrorCheck(
        cudaMemcpy(d_result, &init_val, sizeof(float), cudaMemcpyHostToDevice));
    max_kernel(d_data, d_result, N);
  } else if (operation == "avg" || operation == "mean") {
    avg_kernel(d_data, d_result, N);
  } else if (operation == "median") {
    bitonic_sort(d_data, N);
    median_kernel(d_data, d_result, N);
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
