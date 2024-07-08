
typedef void (*MatMulFuncPtr)(const float* A, const float* B, float* C, int m,
                              int n, int k);

// ----------------- Test Function -----------------
float* convert_2d_to_1d(float** array_2d, int rows, int cols) {
  // cout << *(array_2d[0]) << endl;
  float* result = new float[rows * cols];
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      float* addr = (float*)array_2d + i * cols + j;
      result[i * cols + j] = *addr;
      // result[i * cols + j] = array_2d[i][j];
    }
  }
  return result;
}

void test_matmul1(MatMulFuncPtr matmul_func) {
  std::cout << "Test 1" << std::endl;

  int m = 2, n = 3, k = 2;
  float A_2d[2][3] = {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
  float B_2d[3][2] = {{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}};

  float* A = convert_2d_to_1d(reinterpret_cast<float**>(A_2d), m, n);
  float* B = convert_2d_to_1d(reinterpret_cast<float**>(B_2d), n, k);
  float* C = new float[m * k]{0.0f};

  matmul_func(A, B, C, m, n, k);

  float C_expected[4] = {22.0f, 28.0f, 49.0f, 64.0f};

  // Assert that C matches C_expected
  for (int i = 0; i < m * k; ++i) {
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

  float* A = convert_2d_to_1d(reinterpret_cast<float**>(A_2d), m, n);
  float* B = convert_2d_to_1d(reinterpret_cast<float**>(B_2d), n, k);
  float* C = new float[m * k]{0.0f};

  matmul_func(A, B, C, m, n, k);

  float C_expected[4] = {-10.0f, 12.0f, 19.0f, -24.0f};

  // Assert that C matches C_expected
  for (int i = 0; i < m * k; ++i) {
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