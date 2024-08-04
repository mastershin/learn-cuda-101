#pragma once

#include <cuda_runtime.h>
#include <stdio.h>

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <tuple>

using namespace cv;
using namespace std;

// CUDA stub functions
__host__ void kernel_stub_applySimpleLinearBlurFilter(uchar* r, uchar* g,
                                                      uchar* b, int rows,
                                                      int columns,
                                                      int threadsPerBlock);

// CPU functions
__host__ float compareColorImages(uchar* r0, uchar* g0, uchar* b0, uchar* r1,
                                  uchar* g1, uchar* b1, int rows, int columns);
__host__ void allocateDeviceMemory(int rows, int columns);
__host__ void executeKernel(uchar* r, uchar* g, uchar* b, int rows, int columns,
                            int threadsPerBlock);
__host__ void cleanUpDevice();
__host__ std::tuple<string, string, int> parseCommandLineArguments(
    int argc, char* argv[]);
__host__ std::tuple<int, int, uchar*, uchar*, uchar*> readImageFromFile(
    std::string inputFile);
__host__ std::tuple<uchar*, uchar*, uchar*> applyBlurKernel(
    std::string inputImage);
int main(int argc, char* argv[]);
