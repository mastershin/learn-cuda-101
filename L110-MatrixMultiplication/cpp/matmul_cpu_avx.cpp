#include <immintrin.h>
#include <chrono>
#include <cmath>    // For std::fabs
#include <cstdlib>  // For atoi
#include <iostream>
#include <string>
#include <tuple>  // For std::tuple

#define LOOP 200
#define TOLERANCE 1e-5  // Tolerance for floating-point comparison

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
    for (int j = 0; j < n; ++j) {
      C[i * n + j] = 0.0f;
      for (int p = 0; p < k; ++p) {
        C[i * n + j] += A[i * k + p] * B[p * n + j];
      }
    }
  }
}

// AVX512 matrix multiplication
void matmul_AVX512(const float* A, const float* B, float* C, int m, int n,
                   int k) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      __m512 sum = _mm512_setzero_ps();
      for (int p = 0; p < k; p += 16) {
        __m512 a = _mm512_loadu_ps(&A[i * k + p]);
        __m512 b = _mm512_set1_ps(B[p * n + j]);
        sum = _mm512_fmadd_ps(a, b, sum);
      }
      // Horizontally add the elements in sum
      float temp[16];
      _mm512_storeu_ps(temp, sum);
      C[i * n + j] = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] +
                     temp[6] + temp[7] + temp[8] + temp[9] + temp[10] +
                     temp[11] + temp[12] + temp[13] + temp[14] + temp[15];
    }
  }
}

// AVX2 matrix multiplication
void matmul_AVX2(const float* A, const float* B, float* C, int m, int n,
                 int k) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      __m256 sum = _mm256_setzero_ps();
      for (int p = 0; p < k; p += 8) {
        __m256 a = _mm256_loadu_ps(&A[i * k + p]);
        __m256 b = _mm256_set1_ps(B[p * n + j]);
        sum = _mm256_fmadd_ps(a, b, sum);
      }
      // Horizontally add the elements in sum
      float temp[8];
      _mm256_storeu_ps(temp, sum);
      C[i * n + j] = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] +
                     temp[6] + temp[7];
    }
  }
}

// AVX matrix multiplication
void matmul_AVX(const float* A, const float* B, float* C, int m, int n, int k) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      __m128 sum = _mm_setzero_ps();
      for (int p = 0; p < k; p += 4) {
        __m128 a = _mm_loadu_ps(&A[i * k + p]);
        __m128 b = _mm_set1_ps(B[p * n + j]);
        sum = _mm_add_ps(sum, _mm_mul_ps(a, b));
      }
      // Horizontally add the elements in sum
      float temp[4];
      _mm_storeu_ps(temp, sum);
      C[i * n + j] = temp[0] + temp[1] + temp[2] + temp[3];
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
    A[i] = (i % 2 == 0) ? 1.0f : -1.0f;
  }

  for (int i = 0; i < sizeB; ++i) {
    B[i] = (i % 2 == 0) ? -1.0f : 1.0f;
  }
}

void matmul(const float* A, const float* B, float* C, int m, int n, int k) {
  // Choose the best available function based on CPU capabilities
  // Runtime selection
  if (__builtin_cpu_supports("avx512f")) {
    matmul_AVX512(A, B, C, m, n, k);
  } else if (__builtin_cpu_supports("avx2")) {
    matmul_AVX2(A, B, C, m, n, k);
  } else if (__builtin_cpu_supports("avx")) {
    matmul_AVX(A, B, C, m, n, k);
  } else {
    matmul_cpu(A, B, C, m, n, k);
  }
}

void get_small_matrix_size(int& m, int& n, int& k) {
  m = 200;
  n = 150;
  k = 100;
}

void get_medium_matrix_size(int& m, int& n, int& k) {
  m = 500;
  n = 300;
  k = 200;
}

void get_large_matrix_size(int& m, int& n, int& k) {
  m = 4096;
  n = 1024;
  k = 1024;
}

void log_cpu_features() {
  if (__builtin_cpu_supports("avx512f")) {
    std::cout << "AVX-512 supported" << std::endl;
  }
  if (__builtin_cpu_supports("avx2")) {
    std::cout << "AVX2 supported" << std::endl;
  }
  if (__builtin_cpu_supports("avx")) {
    std::cout << "AVX supported" << std::endl;
  }
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

bool verify_result(const float* C1, const float* C2, int m, int n) {
  for (int i = 0; i < m * n; ++i) {
    if (std::fabs(C1[i] - C2[i]) > TOLERANCE) {
      return false;
    }
  }
  return true;
}

int main(int argc, char* argv[]) {

  auto [m, n, k] = parse_command_args(argc, argv);

  std::cout << "Matrix Multiplication: A(" << m << "x" << k << ") * B(" << k
            << "x" << n << ") = C(" << m << "x" << n << ")" << std::endl;

  log_cpu_features();

  // Allocate memory for matrices A, B, and C

  float *A, *B, *C, *C_cpu, *C_avx512, *C_avx2;

  // Initialize vectors a and b
  initialize_data(A, B, C, m, n, k);
  C_cpu = new float[m * n]{0.0f};
  C_avx512 = new float[m * n]{0.0f};
  C_avx2 = new float[m * n]{0.0f};

  // Perform CPU matrix multiplication for verification
  matmul_cpu(A, B, C_cpu, m, n, k);

  // Verify AVX512 result against CPU result
  if (__builtin_cpu_supports("avx512f")) {
    matmul_AVX512(A, B, C_avx512, m, n, k);
    if (!verify_result(C_cpu, C_avx512, m, n)) {
      std::cerr << "Verification failed for AVX512" << std::endl;
      delete[] A;
      delete[] B;
      delete[] C;
      delete[] C_cpu;
      delete[] C_avx512;
      delete[] C_avx2;
      return 1;
    }
  }

  // Verify AVX2 result against CPU result
  if (__builtin_cpu_supports("avx2")) {
    matmul_AVX2(A, B, C_avx2, m, n, k);
    if (!verify_result(C_cpu, C_avx2, m, n)) {
      std::cerr << "Verification failed for AVX2" << std::endl;
      delete[] A;
      delete[] B;
      delete[] C;
      delete[] C_cpu;
      delete[] C_avx512;
      delete[] C_avx2;
      return 1;
    }
  }
  std::cout << "Verification passed" << std::endl;

  auto start_cpu = now();

  for (int i = 0; i < LOOP; i++) {
    std::cout << "." << std::flush;
    matmul(A, B, C, m, n, k);
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
  delete[] C_cpu;
  delete[] C_avx512;
  delete[] C_avx2;

  return 0;
}
