/*
TODO: Fix the OpenCL code to work on Windows, Linux, and Mac.

## Linux
```
sudo apt-get install opencl-headers
```

## Mac Compilation
Apple M1 Macs support OpenCL 1.2,
but Apple has deprecated the API and recommends migrating to Metal and Metal Performance Shaders.
*/
#include <iostream>
#include <vector>
#include <chrono>

#include <cl/opencl.hpp>

#define SIZE (1000 * 1000 * 200)
#define LOOP 20

using namespace std;
using time_point = std::chrono::time_point<std::chrono::high_resolution_clock>;

time_point now()
{
    return std::chrono::high_resolution_clock::now();
}

auto time_diff(time_point a, time_point b)
{
    return chrono::duration_cast<chrono::duration<double>>(b - a);
}

void initialize_data(vector<float> &x, vector<float> &y, int size)
{
    for (int i = 0; i < size; i++)
    {
        x[i] = i;
        y[i] = i * 2;
    }
}

int main()
{
    vector<float> x(SIZE);
    vector<float> y(SIZE);
    vector<float> out(SIZE, 0.0f);
    float a = 1.0f;
    initialize_data(x, y, SIZE);

    try {
        // Get platform and device
        vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.empty()) {
            std::cerr << "No platforms found. Check OpenCL installation!" << std::endl;
            return -1;
        }
        cl::Platform default_platform = platforms[0];

        vector<cl::Device> devices;
        default_platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
        if (devices.empty()) {
            std::cerr << "No devices found. Check OpenCL installation!" << std::endl;
            return -1;
        }
        cl::Device default_device = devices[0];

        // Create context
        cl::Context context(default_device);

        // Create program
        cl::Program program(context, 
            "__kernel void vector_add(const float a, __global const float *x, __global const float *y, __global float *out) {"
            "   int id = get_global_id(0);"
            "   out[id] = a * x[id] + y[id];"
            "}");

        // Build program
        program.build(default_device);
        if (program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(default_device) != CL_BUILD_SUCCESS) {
            std::cerr << "Error building program: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << std::endl;
            return -1;
        }

        // Create buffers
        cl::Buffer buffer_a(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float), &a);
        cl::Buffer buffer_x(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, x.size() * sizeof(float), x.data());
        cl::Buffer buffer_y(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, y.size() * sizeof(float), y.data());
        cl::Buffer buffer_out(context, CL_MEM_WRITE_ONLY, out.size() * sizeof(float));

        // Create command queue
        cl::CommandQueue queue(context, default_device);

        // Create kernel
        cl::Kernel kernel(program, "vector_add");

        // Set kernel arguments
        kernel.setArg(0, buffer_a);
        kernel.setArg(1, buffer_x);
        kernel.setArg(2, buffer_y);
        kernel.setArg(3, buffer_out);

        auto start_cpu = now();
        for (int i = 0; i < LOOP; i++) {
            std::cout << "." << std::flush;

            // Execute the kernel
            cl::NDRange global_work_size(SIZE);
            queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_work_size, cl::NullRange);

            queue.finish(); // Ensure kernel execution is complete
        }
        auto end_cpu = now();
        auto cpu_duration = time_diff(start_cpu, end_cpu);

        std::cout << std::endl;
        std::cout << "CPU time: " << cpu_duration.count() << " seconds" << std::endl;

        // Read results from the device
        queue.enqueueReadBuffer(buffer_out, CL_TRUE, 0, out.size() * sizeof(float), out.data());

        float sum = 0.0f;
        for (int i = 0; i < SIZE; ++i) {
            sum += out[i];
        }
        std::cout << "Sum: " << sum << std::endl;

    } catch (cl::Error &err) {
        std::cerr << "ERROR: " << err.what() << "(" << err.err() << ")" << std::endl;
        return -1;
    }

    return 0;
}