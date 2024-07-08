/*
Implements GPU based Merge Sort.
*/
#include "merge_sort.h"

#define USE_PRINT 0

#define min(a, b) (a < b ? a : b)
// Based on https://github.com/kevin-albert/cuda-mergesort/blob/master/mergesort.cu

__host__ std::tuple<dim3, dim3, int> parseCommandLineArguments(int argc,
                                                               char** argv) {
  int numElements = 32;
  dim3 threadsPerBlock;
  dim3 blocksPerGrid;

  threadsPerBlock.x = 32;
  threadsPerBlock.y = 1;
  threadsPerBlock.z = 1;

  blocksPerGrid.x = 8;
  blocksPerGrid.y = 1;
  blocksPerGrid.z = 1;

  for (int i = 1; i < argc; i++) {
    if (argv[i][0] == '-' && argv[i][1] && !argv[i][2]) {
      char arg = argv[i][1];
      unsigned int* toSet = 0;
      switch (arg) {
        case 'x':
          toSet = &threadsPerBlock.x;
          break;
        case 'y':
          toSet = &threadsPerBlock.y;
          break;
        case 'z':
          toSet = &threadsPerBlock.z;
          break;
        case 'X':
          toSet = &blocksPerGrid.x;
          break;
        case 'Y':
          toSet = &blocksPerGrid.y;
          break;
        case 'Z':
          toSet = &blocksPerGrid.z;
          break;
        case 'n':
          i++;
          numElements = std::stoi(argv[i]);
          break;
      }
      if (toSet) {
        i++;
        *toSet = (unsigned int)strtol(argv[i], 0, 10);
      }
    }
  }
  return {threadsPerBlock, blocksPerGrid, numElements};
}

__host__ long* generateRandomLongArray(int numElements) {
  long* randomLongs;
  cudaError_t err =
      cudaMallocHost((void**)&randomLongs, numElements * sizeof(long));
  if (err != cudaSuccess) {
    std::cerr << "Failed to allocate pinned host memory (error code "
              << cudaGetErrorString(err) << ")!\n";
    return nullptr;
  }

  srand(static_cast<unsigned int>(time(nullptr)));

  for (int i = 0; i < numElements; ++i) {
    randomLongs[i] = static_cast<long>(rand());

    // mod 10, for debug purpose
    //        randomLongs[i] = static_cast<long>(rand() % 10);
  }

  return randomLongs;
}

__host__ void printHostMemory(long* host_mem, int num_elments) {
  // Output results
  for (int i = 0; i < num_elments; i++) {
    printf("%ld ", host_mem[i]);
  }
  printf("\n");
}

__host__ int main(int argc, char** argv) {

  auto [threadsPerBlock, blocksPerGrid, numElements] =
      parseCommandLineArguments(argc, argv);

  long* data = generateRandomLongArray(numElements);

  printf("Unsorted data: ");
  printHostMemory(data, numElements);

  data = mergesort(data, numElements, threadsPerBlock, blocksPerGrid);

  printf("Sorted data: ");
  printHostMemory(data, numElements);

  cudaFree(data);
}

__host__ std::tuple<long*, long*, dim3*, dim3*> allocateMemory(
    long* input, int numElements) {
  // Actually allocate the two arrays
  int size = numElements * sizeof(long);
  long *D_data, *D_swp;
  cudaMallocManaged(&D_data, size);
  cudaMallocManaged(&D_swp, size);

  // Copy from our input list into the first array
  // Assuming there is an input list `inputList` available in the scope
  cudaMemcpy(D_data, input, size, cudaMemcpyHostToDevice);
  cudaMemcpy(D_swp, input, size, cudaMemcpyHostToDevice);

  // Allocate memory for dim3 structures
  dim3 *D_threads, *D_blocks;
  cudaMalloc(&D_threads, sizeof(dim3));
  cudaMalloc(&D_blocks, sizeof(dim3));

  return {D_data, D_swp, D_threads, D_blocks};
}

__host__ long* mergesort(long* data, long numElements, dim3 threadsPerBlock,
                         dim3 blocksPerGrid) {

  auto [D_data, D_swp, D_threads, D_blocks] = allocateMemory(data, numElements);
  //auto[D_data, D_swp, D_threads, D_blocks] = cpu_allocateMemory(data, numElements);

  long* A = D_data;
  long* B = D_swp;

  long nThreads = threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z *
                  blocksPerGrid.x * blocksPerGrid.y * blocksPerGrid.z;

#if USE_PRINT
  printf("nThreads=%ld\n", nThreads);
#endif

  // Slice up the list and give pieces of it to each thread, letting the pieces grow
  // bigger and bigger until the whole list is sorted
  //
  for (int width = 2; width < (numElements << 1); width <<= 1) {
    long slices = numElements / ((nThreads)*width) + 1;

#if USE_PRINT
    printf("launching gpu kernel: width=%d slices=%ld numElements=%ld\n", width,
           slices, numElements);
#endif

    // Actually call the kernel
    gpu_mergesort<<<blocksPerGrid, threadsPerBlock>>>(
        A, B, numElements, width, slices, D_threads, D_blocks);
    // Switch the input / output arrays instead of copying them around
    long* tmp = A;
    A = B;
    B = tmp;
    //        cudaMemcpy(A, B, numElements * sizeof(long), cudaMemcpyDeviceToDevice);
  }

  cudaDeviceSynchronize();

#if USE_PRINT
  printf("A: ");
  printHostMemory(A, numElements);
  printf("B: ");
  printHostMemory(B, numElements);
#endif

  cudaMemcpy(data, A, numElements * sizeof(long), cudaMemcpyDeviceToHost);

  // Free the GPU memory
  cudaFree(D_data);
  cudaFree(D_swp);
  cudaFree(D_threads);
  cudaFree(D_blocks);

  return data;
}

// GPU helper function
// calculate the id of the current thread
__device__ unsigned int getIdx(dim3* threads, dim3* blocks) {
  int x;
  //return threadIdx.x +
  //       threadIdx.y * (x  = threads->x) +
  //       threadIdx.z * (x *= threads->y) +
  //       blockIdx.x  * (x *= threads->z) +
  //       blockIdx.y  * (x *= blocks->z) +
  //       blockIdx.z  * (x *= blocks->y);

  return blockIdx.x * blockDim.x + threadIdx.x;
}

#if USE_PRINT

__device__ void print(const char* header, const long* arr, long start,
                      long end) {
  printf("%s=", header);
  for (int i = start; i < end; i++) {
    printf("%ld ", arr[i]);
  }
  printf("\n");
}

#endif

//
// Perform a full mergesort on our section of the data.
//
__global__ void gpu_mergesort(const long* source, long* dest, long size,
                              long width, long slices, dim3* threads,
                              dim3* blocks) {

  unsigned int idx = getIdx(threads, blocks);
  if (idx >= size)
    return;

  // Initialize the variables
  long start = width * idx * slices;
  long middle;
  long end;

  if (start >= size)
    return;

#if USE_PRINT
  printf("gpu_mergesort(): start=%ld\n", start);
#endif

  for (long slice = 0; slice < slices; slice++) {
    // Break from loop when the start variable is >= size of the input array
    if (start >= size)
      break;

    // Set middle to be minimum of middle index (start index plus 1/2 width) and the size of the input array
    middle = min(start + width / 2, size);

    // Set end to the minimum of the end index (start index plus the width of the current data window) and the size of the input array
    end = min(start + width, size);

#if USE_PRINT
    print("1. source", source, start, end);
    print("1. dest", dest, start, end);
#endif
    // Perform bottom up merge
    gpu_bottomUpMerge(source, dest, start, middle, end);

#if USE_PRINT
    print("2. source", source, start, end);
    print("2. dest", dest, start, end);
#endif

    start += width;
  }
}

__device__ void gpu_bottomUpMerge(const long* source, long* dest, long start,
                                  long middle, long end) {
  long i = start;
  long j = middle;
  long k = start;

#if USE_PRINT
  printf("start=%ld middle=%ld end=%ld\n", start, middle, end);
  print("source", source, start, end);
#endif
  // Merge the two halves into dest array
  while (i < middle && j < end) {
    if (source[i] <= source[j]) {

#if USE_PRINT
      printf("dest[%ld]=source[%ld] : %ld \n", k, i, source[i]);
#endif
      dest[k++] = source[i++];
    } else {
#if USE_PRINT
      printf("dest[%ld]=source[%ld] : %ld \n", k, j, source[j]);
#endif
      dest[k++] = source[j++];
    }
  }

  // Copy the remaining elements from the left half, if any
  while (i < middle) {
#if USE_PRINT
    printf("left half: dest[%ld]=%ld\n", k, source[i]);
#endif
    dest[k++] = source[i++];
  }
#if USE_PRINT
  print("dest", dest, start, end);
#endif

  // Copy the remaining elements from the right half, if any
  while (j < end) {
#if USE_PRINT
    printf("right half: dest[%ld]=%ld\n", k, source[j]);
#endif
    dest[k++] = source[j++];
  }

#if USE_PRINT
  print("dest", dest, start, end);
#endif
}
