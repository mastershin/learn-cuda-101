#pragma once
void matmul_avx_auto(const float* A, const float* B, float* C, int m, int n,
                     int k);
void matmul_AVX512(const float* A, const float* B, float* C, int m, int n,
                   int k);
void matmul_AVX2(const float* A, const float* B, float* C, int m, int n, int k);
void matmul_AVX(const float* A, const float* B, float* C, int m, int n, int k);