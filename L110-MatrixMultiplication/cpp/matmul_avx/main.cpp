/*
Multiplies Matrices A, B, C with sizes m x k, k x n, m x n)

A x B = C
*/

#include <cassert>
#include <chrono>
#include <iostream>
#include <tuple>
#include <cmath>

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#elif defined(__linux__) || defined(__APPLE__)
#include <stdlib.h>
#include <sys/mman.h>
#endif

#include "avx.h"
#include "matmul_cpu_avx.h"
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
      A[i] = (float)i;  // Assign sequential float values
    else
      A[i] = (float)-i;
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
      std::cerr << C1[i] << " vs " << C2[i] << " at i=" << i << std::endl;
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

void _verify_matmul_avx(int m, int n, int k) {
  // Allocate memory for matrices A, B, and C
  float *A, *B, *C, *C_cpu;

  // Initialize vectors a and b
  initialize_data(A, B, C, m, n, k);
  C_cpu = new float[m * n]{0.0f};

  // Perform CPU matrix multiplication for verification
  matmul_cpu(A, B, C_cpu, m, n, k);

  // Verify AVX512 result against CPU result
  // if (__builtin_cpu_supports("avx512f")) {
  if (detect_avx512()) {
    float* C_avx512 = new float[m * n]{0.0f};
    matmul_AVX512(A, B, C_avx512, m, n, k);
    if (!verify_result(C_cpu, C_avx512, m, n, k)) {
      std::cerr << "Verification failed for AVX-512" << std::endl;
      delete[] A;
      delete[] B;
      delete[] C;
      delete[] C_cpu;
      delete[] C_avx512;
      exit(1);
    }
    delete[] C_avx512;
    std::cout << "Verification Successful for AVX-512" << std::endl;
  }

  // Verify AVX2 result against CPU result
  //if (__builtin_cpu_supports("avx2")) {
  if (detect_avx2()) {
    float* C_avx2 = new float[m * n]{0.0f};
    matmul_AVX2(A, B, C_avx2, m, n, k);
    if (!verify_result(C_cpu, C_avx2, m, n, k)) {
      std::cerr << "Verification failed for AVX2 (256 bit)" << std::endl;
      delete[] A;
      delete[] B;
      delete[] C;
      delete[] C_cpu;
      delete[] C_avx2;
      exit(1);
    }
    delete[] C_avx2;
    std::cout << "Verification Successful for AVX2 (256 bit)" << std::endl;
  }

  // Verify AVX (128 bit) result against CPU result
  //if (__builtin_cpu_supports("avx2")) {
  if (detect_avx()) {
    float* C_avx = new float[m * n]{0.0f};
    matmul_AVX2(A, B, C_avx, m, n, k);
    if (!verify_result(C_cpu, C_avx, m, n, k)) {
      std::cerr << "Verification failed for AVX (128 bit)" << std::endl;
      delete[] A;
      delete[] B;
      delete[] C;
      delete[] C_cpu;
      delete[] C_avx;
      exit(1);
    }
    delete[] C_avx;
    std::cout << "Verification Successful for AVX (128 bit)" << std::endl;
  }

  std::cout << "Verification passed" << std::endl;
}

int do_matmul_avx_best(int m, int n, int k) {

  // Allocate memory for matrices A, B, and C
  float *A, *B, *C;

  // Initialize vectors a and b
  initialize_data(A, B, C, m, n, k);

  auto start_cpu = now();

  for (int i = 0; i < LOOP; i++) {
    std::cout << "." << std::flush;
    matmul_avx_auto(A, B, C, m, n, k);
  }
  auto end_cpu = now();
  std::chrono::duration<double> duration = end_cpu - start_cpu;

  std::cout << std::endl;
  std::cout << "CPU time: " << duration.count() << " seconds" << std::endl;

  float sum = 0.0f;
  for (int i = 0; i < m * n; ++i) {
    sum += C[i];
  }
  std::cout << "Sum: " << sum << std::endl;

  // Clean up
  delete[] A;
  delete[] B;
  delete[] C;

  return 0;
}

int do_matmul_avx(std::string avx_type, int m, int n, int k) {

  // Allocate memory for matrices A, B, and C
  float *A, *B, *C;

  // Initialize vectors a and b
  initialize_data(A, B, C, m, n, k);

  auto start_cpu = now();

  std::cout << avx_type << ":";
  for (int i = 0; i < LOOP; i++) {
    std::cout << "." << std::flush;
    if (avx_type == "avx") {
      matmul_AVX(A, B, C, m, n, k);
    } else if (avx_type == "avx2") {
      matmul_AVX2(A, B, C, m, n, k);
    } else if (avx_type == "avx512") {
      matmul_AVX512(A, B, C, m, n, k);
    }
  }
  auto end_cpu = now();
  std::chrono::duration<double> duration = end_cpu - start_cpu;

  std::cout << std::endl;
  std::cout << avx_type << " CPU time: " << duration.count() << " seconds"
            << std::endl;

  /*
  float sum = 0.0f;
  for (int i = 0; i < m * k; ++i) {
    sum += C[i];
  }
  std::cout << avx_type << " Sum: " << sum << std::endl;
  */

  // Clean up
  delete[] A;
  delete[] B;
  delete[] C;

  return 0;
}

int main(int argc, char* argv[]) {
  print_avx_features();

  auto [m, n, k] = parse_command_args(argc, argv);
  std::cout << "Matrix Multiplication: A(" << m << "x" << k << ") * B(" << k
            << "x" << n << ") = C(" << m << "x" << n << ")" << std::endl;

  std::cout << "Testing matmul_cpu" << std::endl;
  test_matmul(matmul_cpu);

  if (detect_avx()) {
    std::cout << "Testing matmul_AVX" << std::endl;
    test_matmul(matmul_AVX);
  } else {
    std::cout << "Skipping AVX (128 bit) Testing..." << std::endl;
  }

  if (detect_avx2()) {
    std::cout << "Testing matmul_AVX2" << std::endl;
    test_matmul(matmul_AVX2);
  } else {
    std::cout << "Skipping AVX2 (256 bit) Testing..." << std::endl;
  }

  if (detect_avx512()) {
    std::cout << "Testing matmul_AVX512" << std::endl;
    test_matmul(matmul_AVX512);
  } else {
    std::cout << "Skipping AVX-512 Testing..." << std::endl;
  }

  // one first testing with AVX vs non-AVX results
  _verify_matmul_avx(m, n, k);

  if (detect_avx512()) {
    do_matmul_avx("avx512", m, n, k);
  }
  if (detect_avx2()) {
    do_matmul_avx("avx2", m, n, k);
  }
  if (detect_avx()) {
    do_matmul_avx("avx", m, n, k);
  }

  return 0;
}