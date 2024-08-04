#include "linear_filter.hpp"

using namespace std;

__host__ float compareColorImages(uchar* r0, uchar* g0, uchar* b0, uchar* r1,
                                  uchar* g1, uchar* b1, int rows, int columns) {
  cout << "Comparing actual and test pixel arrays\n";
  int numImagePixels = rows * columns;
  int imagePixelDifference = 0.0;

  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < columns; ++c) {
      uchar image0R = r0[r * rows + c];
      uchar image0G = g0[r * rows + c];
      uchar image0B = b0[r * rows + c];
      uchar image1R = r1[r * rows + c];
      uchar image1G = g1[r * rows + c];
      uchar image1B = b1[r * rows + c];
      imagePixelDifference +=
          ((abs(image0R - image1R) + abs(image0G - image1G) +
            abs(image0B - image1B)) /
           3);
    }
  }

  float meanImagePixelDifference = imagePixelDifference / numImagePixels;
  float scaledMeanDifferencePercentage = (meanImagePixelDifference / 255);
  printf("meanImagePixelDifference: %f scaledMeanDifferencePercentage: %f\n",
         meanImagePixelDifference, scaledMeanDifferencePercentage);
  return scaledMeanDifferencePercentage;
}

__host__ void executeKernel(uchar* r, uchar* g, uchar* b, int rows, int columns,
                            int threadsPerBlock) {
  cout << "Executing kernel\n";

  // Define block dimensions based on a fixed configuration, e.g., 16x16
  //dim3 threadsPerBlockDim(threadsPerBlock, threadsPerBlock, 1);
  //dim3 blocksPerGridDim((columns + threadsPerBlockDim.x - 1) / threadsPerBlockDim.x,
  //                      (rows + threadsPerBlockDim.y - 1) / threadsPerBlockDim.y, 1);

  // Calculate shared memory size, including halo for boundary elements
  //int sharedMemSize = (threadsPerBlockDim.x + 2) * threadsPerBlockDim.y * 3 * sizeof(uchar);
  //kernel_applySimpleLinearBlurFilter<<<blocksPerGridDim, threadsPerBlock, sharedMemSize>>>(r, g, b);

  kernel_stub_applySimpleLinearBlurFilter(r, g, b, rows, columns,
                                          threadsPerBlock);

  cudaError_t err = cudaGetLastError();

  if (err != cudaSuccess) {
    fprintf(stderr,
            "Failed to launch applySimpleLinearBlurFilter kernel (error code "
            "%s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

// Reset the device and exit
__host__ void cleanUpDevice() {
  cout << "Cleaning CUDA device\n";
  // cudaDeviceReset causes the driver to clean up all state. While
  // not mandatory in normal operation, it is good practice.  It is also
  // needed to ensure correct operation when the application is being
  // profiled. Calling cudaDeviceReset causes all profile data to be
  // flushed before the application exits
  cudaError_t err = cudaDeviceReset();

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to deinitialize the device! error=%s\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

__host__ tuple<string, string, int> parseCommandLineArguments(int argc,
                                                              char* argv[]) {
  cout << "Parsing CLI arguments\n";
  int threadsPerBlock = 256;
  std::string inputImage = "sloth.png";
  std::string outputImage = "grey-sloth.png";

  for (int i = 1; i < argc; i++) {
    std::string option(argv[i]);
    i++;
    std::string value(argv[i]);
    if (option.compare("-i") == 0) {
      inputImage = value;
    } else if (option.compare("-o") == 0) {
      outputImage = value;
    } else if (option.compare("-t") == 0) {
      threadsPerBlock = atoi(value.c_str());
    }
  }
  cout << "inputImage: " << inputImage << " outputImage: " << outputImage
       << " threadsPerBlock: " << threadsPerBlock << "\n";
  return {inputImage, outputImage, threadsPerBlock};
}

__host__ tuple<int, int, uchar*, uchar*, uchar*> readImageFromFile(
    string inputFile) {
  cout << "Reading Image From File\n";
  Mat img = imread(inputFile, IMREAD_COLOR);
  if (img.empty()) {
    cerr << "Error: Could not load image " << inputFile << endl;
    exit(EXIT_FAILURE);
  }

  const int rows = img.rows;
  const int columns = img.cols;
  size_t size = sizeof(uchar) * rows * columns;

  cout << "Rows: " << rows << " Columns: " << columns << "\n";

  uchar *r, *g, *b;
  cudaMallocManaged(&r, size);
  cudaMallocManaged(&g, size);
  cudaMallocManaged(&b, size);

  for (int y = 0; y < rows; ++y) {
    for (int x = 0; x < columns; ++x) {
      Vec3b rgb = img.at<Vec3b>(y, x);
      b[y * columns + x] = rgb.val[0];
      g[y * columns + x] = rgb.val[1];
      r[y * columns + x] = rgb.val[2];
    }
  }

  return {rows, columns, r, g, b};
}

__host__ tuple<uchar*, uchar*, uchar*> applyBlurKernel(string inputImage) {
  cout << "CPU applying kernel\n";
  Mat img = imread(inputImage, IMREAD_COLOR);
  const int rows = img.rows;
  const int columns = img.cols;

  uchar* r = (uchar*)malloc(sizeof(uchar) * rows * columns);
  uchar* g = (uchar*)malloc(sizeof(uchar) * rows * columns);
  uchar* b = (uchar*)malloc(sizeof(uchar) * rows * columns);

  for (int y = 0; y < rows; ++y) {
    for (int x = 1; x < columns - 1; ++x) {
      Vec3b rgb0 = img.at<Vec3b>(y, x - 1);
      Vec3b rgb1 = img.at<Vec3b>(y, x);
      Vec3b rgb2 = img.at<Vec3b>(y, x + 1);
      b[y * columns + x] = (rgb0[0] + rgb1[0] + rgb2[0]) / 3;
      g[y * columns + x] = (rgb0[1] + rgb1[1] + rgb2[1]) / 3;
      r[y * columns + x] = (rgb0[2] + rgb1[2] + rgb2[2]) / 3;
    }
  }

  return {r, g, b};
}

int main(int argc, char* argv[]) {

  auto [inputImage, outputImage, threadsPerBlock] =
      parseCommandLineArguments(argc, argv);

  try {
    auto [rows, columns, r, g, b] = readImageFromFile(inputImage);

    executeKernel(r, g, b, rows, columns, threadsPerBlock);

    Mat colorImage(rows, columns, CV_8UC3);
    vector<int> compression_params;
    compression_params.push_back(IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(9);

    for (int y = 0; y < rows; ++y) {
      for (int x = 0; x < columns; ++x) {
        int index = y * columns + x;
        //colorImage.at<Vec3b>(y,x) = Vec3b(b[y*rows+x], g[y*rows+x], r[y*rows+x]);
        Vec3b& bgr = colorImage.at<Vec3b>(y, x);
        bgr[0] = b[index];
        bgr[1] = g[index];
        bgr[2] = r[index];
        /*
                Vec3b bgr = colorImage.at<Vec3b>(y, x);
                int index = y * columns + x;
                b[index] = bgr[0];
                g[index] = bgr[1];
                r[index] = bgr[2];
*/
      }
    }

    imwrite(outputImage, colorImage, compression_params);

    auto [test_r, test_g, test_b] = applyBlurKernel(inputImage);

    float scaledMeanDifferencePercentage =
        compareColorImages(r, g, b, test_r, test_g, test_b, rows, columns) *
        100;
    cout << "Mean difference percentage: " << scaledMeanDifferencePercentage
         << "\n";

    cleanUpDevice();
  } catch (Exception& error_) {
    cout << "Caught exception: " << error_.what() << endl;
    return 1;
  }
  return 0;
}
