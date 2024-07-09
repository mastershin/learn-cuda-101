#include <immintrin.h>
// #include <algorithm>
#include <numeric>  // std::accumulate

#include "avx.h"

// Function to convert a row-major 1D array to a column-major 1D array
void _transpose(const float* row_major, float* column_major, int rows,
                int cols) {
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      column_major[j * rows + i] = row_major[i * cols + j];
    }
  }
}

// AVX512 matrix multiplication
void matmul_AVX512(const float* A, const float* B_row, float* C, int m, int n,
                   int k) {
  float* B = new float[n * k];
  _transpose(B_row, B, n, k);

  // AVX register size
  const int register_size = 16;

  // Iterate over each row of C (m rows)
  for (int i = 0; i < m; ++i) {
    // Iterate over each column of C (n columns)
    for (int j = 0; j < k; ++j) {
      // Initialize the result to 0

      __m512 sum = _mm512_setzero_ps();

      // Iterate over each element of the row of A and the column of B
      int l = 0;

      for (; l <= n - register_size; l += register_size) {
        // Load the elements of A and B into AVX registers
        __m512 a = _mm512_loadu_ps(A + i * n + l);
        __m512 b = _mm512_loadu_ps(B + j * n + l);

        // Multiply the elements and add to the sum
        sum = _mm512_fmadd_ps(a, b, sum);
      }

      // Handle remaining elements
      float remaining_sum = 0.0f;
      for (; l < n; ++l) {
        std::cout << m << "," << n << "," << k << " i=" << i << " j=" << j
                  << " l=" << l << " - " << i * n + l << ", " << j * n + l
                  << " A=" << A[i * n + l] << " B=" << B[j * n + l]
                  << std::endl;
        remaining_sum += A[i * n + l] * B[j * n + l];
        // std::cout << " remaining_sum=" << remaining_sum << std::endl;
      }

      // Store the result in C
      float temp[register_size];
      _mm512_storeu_ps(temp, sum);

      float total_sum =
          std::accumulate(temp, temp + register_size, remaining_sum);
      //      std::cout << "C at " << i * k + j << " temp=" << temp[0] << "," << temp[1]
      //                << "," << temp[2] << "," << temp[3]
      //                << " remaining_sum=" << remaining_sum << " total=" << total_sum
      //                << std::endl;
      C[i * k + j] = total_sum;
    }
  }
  delete[] B;
}

// Matrix multiplication using AVX2
void matmul_AVX2(const float* A, const float* B_row, float* C, int m, int n,
                 int k) {
  float* B = new float[n * k];
  _transpose(B_row, B, n, k);

  // AVX register size
  const int register_size = 8;

  // Iterate over each row of C (m rows)
  for (int i = 0; i < m; ++i) {
    // Iterate over each column of C (n columns)
    for (int j = 0; j < k; ++j) {
      // Initialize the result to 0
      __m256 sum = _mm256_setzero_ps();

      // Iterate over each element of the row of A and the column of B
      int l = 0;
      for (; l <= n - register_size; l += register_size) {
        // Load the elements of A and B into AVX registers
        __m256 a = _mm256_loadu_ps(A + i * n + l);
        __m256 b = _mm256_loadu_ps(B + j * n + l);

        // Multiply the elements and add to the sum
        sum = _mm256_fmadd_ps(a, b, sum);
      }

      // Handle remaining elements
      float remaining_sum = 0.0f;
      for (; l < n; ++l) {
        //        std::cout << m << "," << n << "," << k << " i=" << i << " j=" << j
        //                  << " l=" << l << " - " << i * n + l << ", " << j * n + l
        //                  << " A=" << A[i * n + l] << " B=" << B[j * n + l]
        //                  << std::endl;
        remaining_sum += A[i * n + l] * B[j * n + l];
        // std::cout << " remaining_sum=" << remaining_sum << std::endl;
      }

      // Store the result in C
      float temp[register_size];
      _mm256_storeu_ps(temp, sum);

      float total_sum =
          std::accumulate(temp, temp + register_size, remaining_sum);
      //      std::cout << "C at " << i * k + j << " temp=" << temp[0] << "," << temp[1]
      //                << "," << temp[2] << "," << temp[3]
      //                << " remaining_sum=" << remaining_sum << " total=" << total_sum
      //                << std::endl;
      C[i * k + j] = total_sum;
    }
  }
  delete[] B;
}

// AVX 128 matrix multiplication
void matmul_AVX(const float* A, const float* B_row, float* C, int m, int n,
                int k) {
  float* B = new float[n * k];
  _transpose(B_row, B, n, k);

  // AVX register size
  const int register_size = 4;

  // Iterate over each row of C (m rows)
  for (int i = 0; i < m; ++i) {
    // Iterate over each column of C (n columns)
    for (int j = 0; j < k; ++j) {
      // Initialize the result to 0
      __m128 sum = _mm_setzero_ps();

      // Iterate over each element of the row of A and the column of B
      int l = 0;
      for (; l <= n - register_size; l += register_size) {
        // Load the elements of A and B into AVX registers
        __m128 a = _mm_loadu_ps(A + i * n + l);
        __m128 b = _mm_loadu_ps(B + j * n + l);

        // Multiply the elements and add to the sum
        sum = _mm_fmadd_ps(a, b, sum);
      }

      // Handle remaining elements
      float remaining_sum = 0.0f;
      for (; l < n; ++l) {
        //        std::cout << m << "," << n << "," << k << " i=" << i << " j=" << j << " l=" << l << " - "
        //                  << i * n + l
        //                  << ", " << j * n + l
        //                  << " A=" << A[i * n + l] << " B=" << B[j * n + l]
        //                  << std::endl;
        remaining_sum += A[i * n + l] * B[j * n + l];
        // std::cout << " remaining_sum=" << remaining_sum << std::endl;
      }

      // Store the result in C
      float temp[register_size];
      _mm_storeu_ps(temp, sum);

      float total_sum =
          std::accumulate(temp, temp + register_size, remaining_sum);
      //std::cout << "C at " << i * k + j << " temp=" << temp[0] << "," << temp[1]
      //<< "," << temp[2] << "," << temp[3]
      //<< " remaining_sum=" << remaining_sum << " total=" << total_sum << std::endl;
      C[i * k + j] = total_sum;
    }
  }

  delete[] B;
}

void matmul_avx_auto(const float* A, const float* B_row, float* C, int m, int n,
                     int k) {

  float* B = new float[n * k];
  _transpose(B_row, B, n, k);

  // Choose the best available function based on CPU capabilities
  // Runtime selection
  if (detect_avx512()) {
    matmul_AVX512(A, B, C, m, n, k);
  } else if (detect_avx2()) {
    matmul_AVX2(A, B, C, m, n, k);
  } else if (detect_avx()) {
    matmul_AVX(A, B, C, m, n, k);
  } else {
    // matmul_cpu(A, B, C, m, n, k);
    std::cerr << "Designed to run AVX only" << std::endl;
  }
  delete[] B;
}
