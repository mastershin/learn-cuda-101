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