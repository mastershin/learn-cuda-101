#include <iostream>
// #include <helper_cuda.h>
#include <sys/time.h>

#include <tuple>
#include <string>
// #include <random>
#include <cstdlib> // for rand and srand
#include <ctime>   // for time

__host__ long* mergesort(long *data, long size, dim3 threadsPerBlock, dim3 blocksPerGrid);
__host__ long *generateRandomLongArray(int numElements);
__host__ std::tuple<long *, long *, dim3 *, dim3 *> allocateMemory(int numElements);
__host__ void printHostMemory(long *host_mem, int num_elments);
__host__ std::tuple<dim3, dim3, int> parseCommandLineArguments(int argc, char **argv);

__global__ void gpu_mergesort(const long* source, long* dest, long size, long width, long slices, dim3* threads, dim3* blocks) ;
__device__ void gpu_bottomUpMerge(const long* source, long* dest, long start, long middle, long end);

