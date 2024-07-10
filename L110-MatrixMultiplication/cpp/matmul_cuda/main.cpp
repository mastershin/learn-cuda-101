/*
Multiplies Matrices A, B, C with sizes m x k, k x n, m x n)

A x B = C

*/

#include <cassert>
#include <chrono>
#include <iostream>
#include <tuple>

#include "matmul_cpu.h"

#include "test_matmul.h"

#define LOOP 200
#define TOLERANCE 1e-5  // Tolerance for floating-point comparison

using namespace std;
using time_point = std::chrono::time_point<std::chrono::high_resolution_clock>;

time_point now() {
  return std::chrono::high_resolution_clock::now();
}

auto time_diff(time_point a, time_point b) {
  // Return Type: std::chrono::duration<double>
  return std::chrono::duration_cast<std::chrono::duration<double>>(b - a);
}

void initialize_data(float*& A, float*& B, float*& C, int& m, int& n, int& k) {
  // Matrix Multiplication: A * B = C

  // Calculate the size of matrices A, B, and C
  int sizeA = m * n;
  int sizeB = n * k;
  int sizeC = m * k;

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

bool verify_result(const float* C1, const float* C2, int m, int n, int k) {
  for (int i = 0; i < m * k; ++i) {
    if (std::fabs(C1[i] - C2[i]) > TOLERANCE) {
      return false;
    }
  }
  return true;
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
    m = std::stoi(argv[1]);
    n = std::stoi(argv[2]);
    k = std::stoi(argv[3]);
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

  test_matmul(matmul_cpu);

  // Allocate memory for matrices A, B, and C
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

  float sum = 0.0;
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