#include <chrono>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

// 1000 * 1000 * 200 (float) --> takes about ~2 GB of memory
#define SIZE 1000 * 1000 * 200
#define LOOP 20

using namespace std;
using time_point = std::chrono::time_point<std::chrono::high_resolution_clock>;

time_point now() {
  return std::chrono::high_resolution_clock::now();
}

auto time_diff(time_point a, time_point b) {
  // Return Type: std::chrono::duration<double>
  return chrono::duration_cast<std::chrono::duration<double>>(b - a);
}

// Function to perform vector addition on a portion of the vectors
void vectorAddThread(const float* x, const float* y, float* out, int start,
                     int end) {
  for (int i = start; i < end; ++i) {
    out[i] = x[i] + y[i];
  }
}

void initialize_data(float* x, float* y, int size) {
  // yields: Avg: -1.34218 on SIZE = 1000 * 1000 * 200
  int v;
  for (int i = 0; i < size; i++) {
    v = i % 10 + 1;
    if (i % 2 == 0) {
      x[i] = v * 2;
      y[i] = -v;
    } else {
      x[i] = v;
      y[i] = -v * 3;
    }
  }
}

// CPU / parallel execution vector addition using threads
void cpuVectorAdd(const float* x, const float* y, float* out, int size,
                  int num_threads = 4) {

  // Create threads
  vector<thread> threads;
  int chunk_size = size / num_threads;

  // Create and start threads
  for (unsigned int i = 0; i < num_threads; ++i) {
    int start = i * chunk_size;
    int end = (i == num_threads - 1) ? size : (i + 1) * chunk_size;
    threads.push_back(thread(vectorAddThread, x, y, out, start, end));
  }

  // Wait for threads to finish
  for (auto& thread : threads) {
    thread.join();
  }
}

unsigned int get_num_threads() {
  return thread::hardware_concurrency();
}

int main() {
  // Allocate memory for vector a and b
  float* x = new float[SIZE];
  float* y = new float[SIZE];
  float* out = new float[SIZE];

  // Initialize vectors a and b
  initialize_data(x, y, SIZE);

  // Get the number of available hardware threads
  unsigned int num_threads = get_num_threads();

  // Adjust if hardware concurrency is 0 (not supported)
  if (num_threads <= 0) {
    num_threads = 1;  // Default to single thread
  }
  cout << "Number of threads: " << num_threads << endl;

  // CPU vector addition
  auto start_cpu = now();

  for (int i = 0; i < LOOP; i++) {
    cout << "." << flush;
    cpuVectorAdd(x, y, out, SIZE, num_threads);  // Threads are launched here
  }
  auto end_cpu = now();
  chrono::duration<double> cpu_duration = time_diff(start_cpu, end_cpu);

  cout << endl;
  cout << "CPU time: " << cpu_duration.count() << " seconds" << std::endl;

  // Calculate the avg using a loop
  float avg = 0.0f;
  for (int i = 0; i < SIZE; ++i) {
    avg += out[i];
  }
  avg /= SIZE;
  cout << "Avg: " << avg << endl;

  // Clean up
  delete[] x;
  delete[] y;
  delete[] out;

  return 0;
}