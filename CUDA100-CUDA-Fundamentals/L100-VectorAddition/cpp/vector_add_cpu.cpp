#include <iostream>
#include <chrono>
#include <ranges>

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
    return chrono::duration_cast<std::chrono::duration<double>>(b - a);
}

// Classic CPU vector addition using for loop (slow)
void cpuVectorAdd(float a, float *x, float *y, float *out, int size)
{
    for (int i = 0; i < size; i++)
    {
        out[i] = x[i] + y[i];
    }
}

void cpuVectorAdd_pointer(float a, float *x, float *y, float *out, int size)
{
    // this function uses pointers
    // for speed gain (~25%), but could be considered unsafe.
    for (int i = 0; i < size; i++)
    {
        *out = *x + *y;
        out++;
    }
}

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
    // Allocate memory for vector a and b
    float a = 2.0;
    float *x = new float[SIZE];
    float *y = new float[SIZE];

    // Initialize vectors a and b
    initialize_data(x, y, SIZE);

    float *out = new float[SIZE]{0.0f};

    // CPU vector addition
    auto start_cpu = now();

    for (int i = 0; i < LOOP; i++)
    {
        cout << "." << flush;
        cpuVectorAdd(a, x, y, out, SIZE);
    }
    auto end_cpu = now();
    chrono::duration<double> cpu_duration = end_cpu - start_cpu;

    cout << endl;
    cout << "CPU time: " << cpu_duration.count() << " seconds" << std::endl;

    // Calculate the sum using a loop
    float sum = 0.0f;
    for (int i = 0; i < SIZE; ++i) {
        sum += out[i];
    }
    cout << "Sum: " << sum << endl;

    // Clean up
    delete[] x;
    delete[] y;
    delete[] out;

    return 0;
}
