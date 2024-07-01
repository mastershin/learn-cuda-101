#include <iostream>
#include <chrono>
#include <immintrin.h>

#define LOOP 200

using time_point = std::chrono::time_point<std::chrono::high_resolution_clock>;

time_point now()
{
    return std::chrono::high_resolution_clock::now();
}
auto time_diff(time_point a, time_point b)
{
    // Return Type: std::chrono::duration<double>
    return std::chrono::duration_cast<std::chrono::duration<double>>(b - a);
}

// Classic CPU matrix multiplication using for loop (slow)
void matmul_cpu(const float *A, const float *B, float *C, int m, int n, int k)
{
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            C[i * n + j] = 0.0f;
            for (int p = 0; p < k; ++p)
            {
                C[i * n + j] += A[i * k + p] * B[p * n + j];
            }
        }
    }
}

// AVX512 matrix multiplication
void matmul_AVX512(const float *A, const float *B, float *C, int m, int n, int k)
{
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            __m512 sum = _mm512_setzero_ps();
            for (int p = 0; p < k; p += 16)
            {
                __m512 a = _mm512_loadu_ps(&A[i * m + p]);
                __m512 b = _mm512_loadu_ps(&B[p * n + j]);
                sum = _mm512_fmadd_ps(a, b, sum);
            }
            _mm512_storeu_ps(&C[i * m + j], sum);
        }
    }
}

// AVX2 matrix multiplication
void matmul_AVX2(const float *A, const float *B, float *C, int m, int n, int k)
{
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            __m256 sum = _mm256_setzero_ps();
            for (int p = 0; p < k; p += 8)
            {
                __m256 a = _mm256_loadu_ps(&A[i * m + p]);
                __m256 b = _mm256_loadu_ps(&B[p * n + j]);
                sum = _mm256_fmadd_ps(a, b, sum);
            }


            _mm256_storeu_ps(&C[i * m + j], sum);
        }
    }
}

// AVX matrix multiplication
void matmul_AVX(const float *A, const float *B, float *C, int m, int n, int k)
{
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            __m128 sum = _mm_setzero_ps();
            for (int p = 0; p < k; p += 4)
            {
                __m128 a = _mm_loadu_ps(&A[i * m + p]);
                __m128 b = _mm_loadu_ps(&B[p * n + j]);
                sum = _mm_add_ps(sum, _mm_mul_ps(a, b));
            }
            _mm_storeu_ps(&C[i * m + j], sum);
        }
    }
}

void initialize_data(float *&A, float *&B, float *&C, int &m, int &n, int &k)
{
    // Matrix Multiplication: A * B = C

    // Calculate the size of matrices A, B, and C
    int sizeA = m * k;
    int sizeB = k * n;
    int sizeC = m * n;

    // Allocate memory for the matrices dynamically
    A = new float[sizeA];
    B = new float[sizeB];
    C = new float[sizeC]{0.0f};
/*
    // Initialize matrices A and B with sequential float numbers
    for (int i = 0; i < sizeA; ++i)
    {
        if (i % 2 == 0)
            A[i] = i + 1.0f; // Assign sequential float values
        else
            A[i] = -i - 1.0f; // Assign sequential float values
    }

    for (int i = 0; i < sizeB; ++i)
    {
        if (i % 2 == 0)
            B[i] = -i - 2.0f; // Assign sequential float values
        else
            B[i] = i + 2.0f; // Assign sequential float values
    }
*/
    // Initialize matrices A and B with sequential float numbers
    for (int i = 0; i < sizeA; ++i) {
        if ( i % 2 == 0)
            A[i] = 1.0f; // Assign sequential float values
        else
            A[i] = -1.0f;
            // A[i] = -i - 1.0f; // Assign sequential float values
    }

    for (int i = 0; i < sizeB; ++i) {
        if ( i % 2 == 0)
            //B[i] = -i - 2.0f;  // Assign sequential float values
            B[i] = -1.0f;  // Assign sequential float values
        else
            B[i] = 1.0f;
            // B[i] = i + 2.0f;  // Assign sequential float values
    }

}

void matmul(const float *A, const float *B, float *C, int m, int n, int k)
{
    // Choose the best available function based on CPU capabilities
    // Runtime selection
    if (__builtin_cpu_supports("avx512f"))
    {
        matmul_AVX512(A, B, C, m, n, k);
    }
    else if (__builtin_cpu_supports("avx2"))
    {
        matmul_AVX2(A, B, C, m, n, k);
    }
    else if (__builtin_cpu_supports("avx"))
    {
        matmul_AVX(A, B, C, m, n, k);
    }
    else
    {
        matmul_cpu(A, B, C, m, n, k);
    }
}

void get_small_matrix_size(int &m, int &n, int &k)
{
    // (200x150, 150x100 --> 200x100)
    // m=200, n=150, k=100 --> sum --> -200 (~1.3 sec on CPU)
    m = 200;
    n = 150;
    k = 100;
}
void get_medium_matrix_size(int &m, int &n, int &k)
{
    // 500x300, 300x200, 500x200 --> sum --> -400 (~13 sec on CPU)
    m = 500;
    n = 300;
    k = 200;
}

void get_large_matrix_size(int &m, int &n, int &k)
{
    // More realistic small-scale LLM size
    // 4096x1024, 1024x1024, 4096x1024 --> sum --> ?
    // int m = 4096, n = 1024, k = 1024;

    m = 4096;
    n = 1024;
    k = 1024;
}

void log_cpu_features()
{
    if (__builtin_cpu_supports("avx512f"))
    {
        std::cout << "AVX-512 supported" << std::endl;
    }
    if (__builtin_cpu_supports("avx2"))
    {
        std::cout << "AVX2 supported" << std::endl;
    }
    if (__builtin_cpu_supports("avx"))
    {
        std::cout << "AVX supported" << std::endl;
    }
}
int main()
{
    log_cpu_features();

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

    for (int i = 0; i < LOOP; i++)
    {
        std::cout << "." << std::flush;
        matmul(A, B, C, m, n, k);
    }
    auto end_cpu = now();
    std::chrono::duration<double> duration = end_cpu - start_cpu;

    std::cout << std::endl;
    std::cout << "CPU time: " << duration.count() << " seconds" << std::endl;

    float sum;
    for (int i = 0; i < m * n; ++i)
    {
        {
            sum += C[i];
        }
    }
    std::cout << "Sum: " << sum << std::endl;

    // Clean up
    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}