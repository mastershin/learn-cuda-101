#include <iostream>
#include <chrono>

#define SIZE 1000 * 1000 * 10
#define LOOP 100

// CPU vector addition
void cpuVectorAdd(int *a, int *b, int *c, int size)
{
    for (int i = 0; i < size; i++)
    {
        c[i] = a[i] + b[i];
    }
}

int main()
{
    int *a = new int[SIZE];
    int *b = new int[SIZE];
    // Allocate memory for vector b
    // Initialize vector b
    for (int i = 0; i < SIZE; i++)
    {
        b[i] = i * 2;
    }
    int *c = new int[SIZE];

    // Initialize vectors a and b
    for (int i = 0; i < SIZE; i++)
    {
        a[i] = i;
        b[i] = i * 2;
    }

    // CPU vector addition
    auto start_cpu = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < LOOP; i++)
    {
        cpuVectorAdd(a, b, c, SIZE);
    }
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_duration = end_cpu - start_cpu;

    std::cout << "CPU time: " << cpu_duration.count() << " seconds" << std::endl;

    // Clean up
    delete[] a;
    delete[] b;
    delete[] c;

    return 0;
}
