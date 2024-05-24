/*
For SIZE 1000 * 1000 * 200, the CPU time is ~5.4 sec on Xeon E5 3 Ghz
*/
#include <iostream>
#include <chrono>

// Required for AVX-512 intrinsics
#include <immintrin.h>

// 1000 * 1000 * 200 (float) --> takes about ~2 GB of memory
#define SIZE 1000 * 1000 * 200
#define LOOP 20

using namespace std;
using time_point = std::chrono::time_point<std::chrono::high_resolution_clock>;

time_point now()
{
    return std::chrono::high_resolution_clock::now();
}
auto time_diff(time_point a, time_point b)
{
    // Return Type: std::chrono::duration<double>
    return chrono::duration_cast<chrono::duration<double>>(b - a);
}

// ----------------- AVX Vector Addition Functions --------------

// Function to perform vector addition on the CPU with AVX
// a *X + Y -> out
template <typename T>
class VectorAdditionAVX
{
private:
    T a = 0.0; // Store the scalar value
    T *x = NULL;
    T *y = NULL;
    T *out = NULL;
    int size; // Store the vector size

    // AVX-512 Implementation
    static void cpuVectorAddAVX512(T a, const T *x, const T *y, T *out, int size)
    {
        int i = 0;
        for (; i <= size - 16; i += 16)
        {
            __m512 vx = _mm512_loadu_ps(x + i);
            __m512 vy = _mm512_loadu_ps(y + i);
            __m512 vout = _mm512_add_ps(vx, vy);
            _mm512_storeu_ps(out + i, vout);
        }
        for (; i < size; ++i)
        {
            out[i] = a * x[i] + y[i];
        }
    }

    // AVX2 Implementation
    static void cpuVectorAddAVX2(T a, const T *x, const T *y, T *out, int size)
    {
        int i = 0;
        for (; i <= size - 8; i += 8)
        {
            __m256 vx = _mm256_loadu_ps(x + i);
            __m256 vy = _mm256_loadu_ps(y + i);
            __m256 vout = _mm256_add_ps(vx, vy);
            _mm256_storeu_ps(out + i, vout);
        }
        for (; i < size; ++i)
        {
            out[i] = a * x[i] + y[i];
        }
    }

    // AVX Implementation
    static void cpuVectorAddAVX(T a, const T *x, const T *y, T *out, int size)
    {
        int i = 0;
        for (; i <= size - 4; i += 4)
        {
            __m128 vx = _mm_loadu_ps(x + i);
            __m128 vy = _mm_loadu_ps(y + i);
            __m128 vout = _mm_add_ps(vx, vy);
            _mm_storeu_ps(out + i, vout);
        }
        for (; i < size; ++i)
        {
            out[i] = a * x[i] + y[i];
        }
    }

    // Scalar Implementation (Fallback)
    static void cpuVectorAddScalar(T a, const T *x, const T *y, T *out, int size)
    {
        for (int i = 0; i < size; ++i)
        {
            out[i] = a * x[i] + y[i];
        }
    }

    // Function to detect the highest available AVX version at runtime
    // Function pointer type for vector add functions
    typedef void (*VectorAddFunc)(T, const T *, const T *, T *, int);

    VectorAddFunc _bestVectorAdd()
    {
        if (__builtin_cpu_supports("avx512f"))
        {
            return cpuVectorAddAVX512;
        }
        else if (__builtin_cpu_supports("avx2"))
        {
            return cpuVectorAddAVX2;
        }
        else if (__builtin_cpu_supports("avx"))
        {
            return cpuVectorAddAVX;
        }
        else
        {
            return cpuVectorAddScalar;
        }
    }

public:
    VectorAdditionAVX(int size) : size(size)
    {
        x = new T[size];
        y = new T[size];
        out = new T[size]{0.0};
    }
    ~VectorAdditionAVX()
    {
        if (x != NULL)
            delete[] x;
        if (y != NULL)
            delete[] y;
        if (out != NULL)
            delete[] out;
    }
    T *get_x() { return x; }
    T *get_y() { return y; }
    T *get_out() { return out; }
    void run()
    {
        VectorAddFunc func = _bestVectorAdd();
        func(a, x, y, out, size);
    }
};

void log_cpu_features()
{
    if (__builtin_cpu_supports("avx512f"))
    {
        cout << "AVX-512 supported" << endl;
    }
    if (__builtin_cpu_supports("avx2"))
    {
        cout << "AVX2 supported" << endl;
    }
    if (__builtin_cpu_supports("avx"))
    {
        cout << "AVX supported" << endl;
    }
}

// ----------------- Main Function -----------------
void initialize_data(float *x, float *y, int size)
{
    for (int i = 0; i < size; i++)
    {
        x[i] = i;
        y[i] = i * 2;
    }
}

int main()
{

    VectorAdditionAVX<float> vectorAdd(SIZE);
    // Initialize vectors a and b
    initialize_data(vectorAdd.get_x(), vectorAdd.get_y(), SIZE);

    log_cpu_features();

    // CPU vector addition
    auto start_cpu = now();

    for (int i = 0; i < LOOP; i++)
    {
        cout << "." << flush;
        vectorAdd.run();
    }
    auto end_cpu = now();
    auto cpu_duration = time_diff(start_cpu, end_cpu);

    cout << endl;
    cout << "CPU time: " << cpu_duration.count() << " seconds" << endl;

    // Calculate the sum using a loop
    float *out = vectorAdd.get_out();
    float sum = 0.0f;
    for (int i = 0; i < SIZE; ++i)
    {
        sum += out[i];
    }
    cout << "Sum: " << sum << endl;

    return 0;
}
