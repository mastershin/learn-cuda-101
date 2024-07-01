#include <chrono>
#include <iostream>
#include <ranges>
#include <cstdlib> // For atoi

// 1000 * 1000 * 200 (float) --> takes about ~2 GB of memory
#define SIZE 1000 * 1000 * 200

using namespace std;
using time_point = std::chrono::time_point<std::chrono::high_resolution_clock>;

time_point now() {
  return std::chrono::high_resolution_clock::now();
}

auto time_diff(time_point a, time_point b) {
  // Return Type: std::chrono::duration<double>
  return chrono::duration_cast<std::chrono::duration<double>>(b - a);
}

// Classic CPU vector addition using for loop (slow)
void cpuVectorAdd(const float* x, const float* y, float* out, int size) {
  for (int i = 0; i < size; i++) {
    out[i] = x[i] + y[i];
  }
}

void cpuVectorAdd_pointer(const float* x, const float* y, float* out, int size) {
  // this function uses pointers
  // for speed gain (~25%), but could be considered unsafe.
  for (int i = 0; i < size; i++) {
    *out = *x + *y;
    out++;
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

int main(int argc, char* argv[]) {
  // Get LOOP value from command line argument
  int loop = 50; // Default value
  if (argc > 1) {
    loop = atoi(argv[1]);
  }
  cout << "LOOP: " << loop << endl;

  // Allocate memory for vector a and b
  float* x = new float[SIZE];
  float* y = new float[SIZE];

  // Initialize vectors a and b
  initialize_data(x, y, SIZE);

  float* out = new float[SIZE];

  // CPU vector addition
  auto start_cpu = now();

  for (int i = 0; i < loop; i++) {
    cout << "." << flush;
    cpuVectorAdd(x, y, out, SIZE);
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
