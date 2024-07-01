#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

#define SIZE 1000 * 1000 * 10
#define BLOCK_SIZE 256
#define LOOP 100

// Utility function to get the current time in seconds
using time_point = std::chrono::time_point<std::chrono::high_resolution_clock>;

time_point now() {
    return std::chrono::high_resolution_clock::now();
}
auto time_diff(time_point a, time_point b) {
    // Return Type: std::chrono::duration<double>
    return std::chrono::duration_cast<std::chrono::duration<double>>(b - a);
}

// GPU kernel for vector addition
__global__ void gpuVectorAdd(float a, int *x, int *y, int *out, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        out[idx] = a * x[idx] + y[idx];
    }
}

// Function to perform vector addition on the GPU
void performVectorAddition(float a, float *x, float *y, float *out, int size)
{
    // Allocate memory for arrays d_a, d_b, and d_c on the device
    int *d_x, *d_y, *d_out;
    cudaMalloc((void **)&d_x, size * sizeof(float));
    cudaMalloc((void **)&d_y, size * sizeof(float));
    cudaMalloc((void **)&d_out, size * sizeof(float));

    // Copy vectors a and b from host to device
    cudaMemcpy(d_x, x, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, y, size * sizeof(float), cudaMemcpyHostToDevice);

    // Calculate the number of blocks needed
    int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Launch the GPU kernel for vector addition
    gpuVectorAdd<<<numBlocks, BLOCK_SIZE>>>(a, d_x, d_y, d_out, size);

    // Wait for GPU to finish execution
    cudaDeviceSynchronize();

    // Copy vector c from device to host
    cudaMemcpy(out, d_out, size * sizeof(int), cudaMemcpyDeviceToHost);

    // Free memory on the device
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_out);
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
    // Allocate memory for arrays a, b, and c on the host
    float a = 2.0;
    float *x = new float[SIZE];
    float *y = new float[SIZE];
    float *out = new float[SIZE];

    // Initialize vectors a and b
    initialize_data(x, y, SIZE);

    // Call the function to perform vector addition on the GPU
    // Start GPU timer
    auto start_gpu = now();

    for (int i = 0; i < LOOP; i++)
    {
        std::cout << "." << std::flush;
        performVectorAddition(a, x, y, out, SIZE);
    }

    // Stop GPU timer
    auto end_gpu = now();
    auto gpu_duration = time_diff(start_gpu, end_gpu);

    // Print GPU execution time
    std::cout << std::endl;
    std::cout << "GPU time: " << gpu_duration.count() << " seconds" << std::endl;

    // Clean up memory on the host
    delete[] x;
    delete[] y;
    delete[] out;

    return 0;
}
