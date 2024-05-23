#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

#define SIZE 1000 * 1000 * 10
#define BLOCK_SIZE 256
#define LOOP 100

// GPU kernel for vector addition
__global__ void gpuVectorAdd(int *a, int *b, int *c, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        c[idx] = a[idx] + b[idx];
    }
}

// Function to perform vector addition on the GPU
void performVectorAddition(int *a, int *b, int *c, int size)
{
    // Allocate memory for arrays d_a, d_b, and d_c on the device
    int *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, size * sizeof(int));
    cudaMalloc((void **)&d_b, size * sizeof(int));
    cudaMalloc((void **)&d_c, size * sizeof(int));

    // Copy vectors a and b from host to device
    cudaMemcpy(d_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size * sizeof(int), cudaMemcpyHostToDevice);

    // Calculate the number of blocks needed
    int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Launch the GPU kernel for vector addition
    gpuVectorAdd<<<numBlocks, BLOCK_SIZE>>>(d_a, d_b, d_c, size);

    // Wait for GPU to finish execution
    cudaDeviceSynchronize();

    // Copy vector c from device to host
    cudaMemcpy(c, d_c, size * sizeof(int), cudaMemcpyDeviceToHost);

    // Free memory on the device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

int main()
{
    // Allocate memory for arrays a, b, and c on the host
    int *a = new int[SIZE];
    int *b = new int[SIZE];
    int *c = new int[SIZE];

    // Initialize vectors a and b
    for (int i = 0; i < SIZE; i++)
    {
        a[i] = i;
        b[i] = i * 2;
    }

    // Call the function to perform vector addition on the GPU
    // Start GPU timer
    auto start_gpu = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < LOOP; i++)
    {
        performVectorAddition(a, b, c, SIZE);
    }

    // Stop GPU timer
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gpu_duration = end_gpu - start_gpu;

    // Print GPU execution time
    std::cout << "GPU time: " << gpu_duration.count() << " seconds" << std::endl;

    // Clean up memory on the host
    delete[] a;
    delete[] b;
    delete[] c;

    return 0;
}
