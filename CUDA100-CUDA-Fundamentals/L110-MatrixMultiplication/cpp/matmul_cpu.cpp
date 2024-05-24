/*
Multiplies Matrices A, B, C with sizes m x k, k x n, m x n)

A x B = C

*/
#include <iostream>
#include <chrono>
#include <cassert>

#define LOOP 200

using namespace std;
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
void matmul_cpu(const float *A, const float *B, float *C, int m, int n, int k) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < k; ++j) {
            C[i * k + j] = 0.0f;
            for (int p = 0; p < n; ++p) {
                C[i * k + j] += A[i * n + p] * B[p * k + j];
            }
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

void get_small_matrix_size(int &m, int &n, int &k) {
    // (200x150, 150x100 --> 200x100)
    // m=200, n=150, k=100 --> sum --> -200 (~1.3 sec on CPU)
    m = 200;
    n = 150;
    k = 100;
}

void get_medium_matrix_size(int &m, int &n, int &k) {
    // 500x300, 300x200, 500x200 --> sum --> -400 (~13 sec on CPU)
    m = 500;
    n = 300;
    k = 200;
}

void get_large_matrix_size(int &m, int &n, int &k) {
    // More realistic small-scale LLM size
    // 4096x1024, 1024x1024, 4096x1024 --> sum --> ?
    // int m = 4096, n = 1024, k = 1024;

    m = 4096;
    n = 1024;
    k = 1024;
}

// ----------------- Test Function -----------------
float* convert_2d_to_1d(float** array_2d, int rows, int cols) {
    // cout << *(array_2d[0]) << endl;
    float* result = new float[rows * cols];
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float* addr = (float*)array_2d + i*cols + j;
            result[i * cols + j] = *addr;
            //result[i * cols + j] = array_2d[i][j];
        }
    }
    return result;
}

void test_matmul1() {
    cout << "Test 1" << endl;

    int m = 2, n = 3, k = 2;
    float A_2d[2][3] = {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
    float B_2d[3][2] = {{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}};

    float* A = convert_2d_to_1d(reinterpret_cast<float**>(A_2d), m, n);
    float* B = convert_2d_to_1d(reinterpret_cast<float**>(B_2d), n, k);
    float* C = new float[m * k]{0.0f};

    matmul_cpu(A, B, C, m, n, k);

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

void test_matmul2() {
    cout << "Test 2" << endl;

    int m = 2, n = 3, k = 2;
    float A_2d[2][3] = {{1.0f, -2.0f, 3.0f}, {-4.0f, 5.0f, -6.0f}};
    float B_2d[3][2] = {{-1.0f, 2.0f}, {-3.0f, 4.0f}, {-5.0f, 6.0f}};

    float* A = convert_2d_to_1d(reinterpret_cast<float**>(A_2d), m, n);
    float* B = convert_2d_to_1d(reinterpret_cast<float**>(B_2d), n, k);
    float* C = new float[m * k]{0.0f};

    matmul_cpu(A, B, C, m, n, k);

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

void test_matmul() {
    test_matmul1();
    test_matmul2();
}

// ----------------- Main Function -----------------

int main()
{
    test_matmul();

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
        matmul_cpu(A, B, C, m, n, k);
    }
    auto end_cpu = now();
    std::chrono::duration<double> duration = end_cpu - start_cpu;

    std::cout << std::endl;
    std::cout << "CPU time: " << duration.count() << " seconds" << std::endl;

    float sum;
    for (int i = 0; i < m * n; ++i) {
        sum += C[i];
    }
    std::cout << "Sum: " << sum << std::endl;
    // printMatrix(C, m, n);

    // Clean up
    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}
