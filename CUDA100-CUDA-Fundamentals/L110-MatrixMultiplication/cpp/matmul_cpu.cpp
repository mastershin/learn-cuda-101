/*
Multiplies Matrices A, B, C with sizes m x k, k x n, m x n)

A x B = C

*/
#include <cassert>
#include <chrono>
#include <iostream>

#include "test_maumul.h"

#define LOOP 200

using namespace std;
using time_point = std::chrono::time_point<std::chrono::high_resolution_clock>;

time_point now() {
  return std::chrono::high_resolution_clock::now();
}
auto time_diff(time_point a, time_point b) {
  // Return Type: std::chrono::duration<double>
  return std::chrono::duration_cast<std::chrono::duration<double>>(b - a);
}

// Classic CPU matrix multiplication using for loop (slow)
void matmul_cpu(const float* A, const float* B, float* C, int m, int n, int k) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < k; ++j) {
      C[i * k + j] = 0.0f;
      for (int p = 0; p < n; ++p) {
        C[i * k + j] += A[i * n + p] * B[p * k + j];
      }
    }
  }
}

void initialize_data(float*& A, float*& B, float*& C, int& m, int& n, int& k) {
  // Matrix Multiplication: A * B = C

  // Calculate the size of matrices A, B, and C
  int sizeA = m * k;
  int sizeB = k * n;
  int sizeC = m * n;

  // Allocate memory for the matrices dynamically
  A = new float[sizeA];
  B = new float[sizeB];
  C = new float[sizeC]{0.0f};

  // Initialize matrices A and B with sequential float numbers
  for (int i = 0; i < sizeA; ++i) {
    if (i % 2 == 0)
      A[i] = i;  // Assign sequential float values
    else
      A[i] = -i;
  }

  for (int i = 0; i < sizeB; ++i) {
    if (i % 3 == 0)
      B[i] = 1.0f;  // Assign sequential float values
    else if (i % 3 == 1)
      B[i] = 2.0f;
    else
      B[i] = 3.0f;
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

// ----------------- Main Function -----------------

int main() {
  test_matmul(matmul_cpu);

  int m, n, k;
  // Allocate memory for matrices A, B, and C

  get_small_matrix_size(m, n, k);
  // get_medium_matrix_size(m, n, k);
  // get_large_matrix_size(m, n, k);

  float *A, *B, *C;

  // Initialize vectors a and b
  initialize_data(A, B, C, m, n, k);

  // CPU vector addition
  auto start_cpu = now();

  for (int i = 0; i < LOOP; i++) {
    std::cout << "." << std::flush;
    matmul_cpu(A, B, C, m, n, k);
  }
  auto end_cpu = now();
  std::chrono::duration<double> duration = end_cpu - start_cpu;

  cout << std::endl;
  cout << "CPU time: " << duration.count() << " seconds" << endl;

  float sum;
  for (int i = 0; i < m * n; ++i) {
    sum += C[i];
  }
  cout << "Sum: " << sum << endl;

  // Clean up
  delete[] A;
  delete[] B;
  delete[] C;

  return 0;
}
