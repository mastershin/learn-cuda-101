#pragma once

typedef void (*MatMulFuncPtr)(const float* A, const float* B, float* C, int m,
                              int n, int k);

// Convert a 2D array to a 1D array in row-major order
template <size_t rows, size_t cols>
void convert_2d_to_1d_row_major(float (&array_2d)[rows][cols],
                                float*& array_1d) {
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      array_1d[i * cols + j] = array_2d[i][j];
    }
  }
}

// Convert a 2D array to a 1D array in column-major order
template <size_t rows, size_t cols>
void convert_2d_to_1d_column_major(float (&array_2d)[rows][cols],
                                   float*& array_1d) {
  for (int j = 0; j < cols; ++j) {
    for (int i = 0; i < rows; ++i) {
      array_1d[j * rows + i] = array_2d[i][j];
    }
  }
}

void test_matmul1(MatMulFuncPtr matmul_func) {
  std::cout << "Test 1" << std::endl;

  int m = 2, n = 3, k = 2;
  float A_2d[2][3] = {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
  float B_2d[3][2] = {{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}};

  float* A = new float[m * n];
  float* B = new float[n * k];
  convert_2d_to_1d_row_major(A_2d, A);
  convert_2d_to_1d_row_major(B_2d, B);
  float* C = new float[m * k]{0.0f};

  matmul_func(A, B, C, m, n, k);

  float C_expected[4] = {22.0f, 28.0f, 49.0f, 64.0f};

  // Assert that C matches C_expected
  for (int i = 0; i < m * k; ++i) {
    if (C[i] != C_expected[i]) {
      std::cerr << C[i] << " vs " << C_expected[i] << " at i=" << i
                << std::endl;
    }
    assert(C[i] == C_expected[i]);
  }

  std::cout << "Test 1 passed." << std::endl;

  delete[] A;
  delete[] B;
  delete[] C;
}

void test_matmul2(MatMulFuncPtr matmul_func) {
  std::cout << "Test 2" << std::endl;

  int m = 2, n = 3, k = 2;
  float A_2d[2][3] = {{1.0f, -2.0f, 3.0f}, {-4.0f, 5.0f, -6.0f}};
  float B_2d[3][2] = {{-1.0f, 2.0f}, {-3.0f, 4.0f}, {-5.0f, 6.0f}};

  float* A = new float[m * n];
  float* B = new float[n * k];
  convert_2d_to_1d_row_major(A_2d, A);
  convert_2d_to_1d_row_major(B_2d, B);
  float* C = new float[m * k]{0.0f};

  matmul_func(A, B, C, m, n, k);

  float C_expected[4] = {-10.0f, 12.0f, 19.0f, -24.0f};

  // Assert that C matches C_expected
  for (int i = 0; i < m * k; ++i) {
    if (C[i] != C_expected[i]) {
      std::cerr << C[i] << " vs " << C_expected[i] << " at i=" << i
                << std::endl;
    }

    assert(C[i] == C_expected[i]);
  }

  std::cout << "Test 2 passed." << std::endl;

  delete[] A;
  delete[] B;
  delete[] C;
}

void test_matmul(MatMulFuncPtr matmul_func) {
  test_matmul1(matmul_func);
  test_matmul2(matmul_func);
}