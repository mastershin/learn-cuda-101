# IDIR=./
# CXX = nvcc

# CXXFLAGS += $(shell pkg-config --cflags --libs opencv4)
# LDFLAGS += $(shell pkg-config --libs --static opencv)

# all: clean build

# build: 
# 	$(CXX) linear_filter.cu --std c++17 `pkg-config opencv --cflags --libs` -o linear_filter.exe -Wno-deprecated-gpu-targets $(CXXFLAGS) -I/usr/local/cuda/include -lcuda

# run:
# 	./linear_filter.exe $(ARGS)

# clean:
# 	rm -f linear_filter.exe

# Define variables
NVCC = nvcc
GCC = g++
CXXFLAGS = --std=c++17 -I/usr/local/cuda/include $(shell pkg-config --cflags opencv4)
LDFLAGS += $(shell pkg-config --libs --static opencv4)

CUDA_FLAGS = -I/usr/local/cuda/include -Xcompiler -fPIC $(shell pkg-config --cflags opencv4)
# CUDA_LIBS = $(shell pkg-config --libs cuda)
CUDA_LIBS = -lcuda
# CUDA architecture flags
ARCH_FLAGS = \
    -gencode arch=compute_60,code=sm_60 \
    -gencode arch=compute_70,code=sm_70 \
    -gencode arch=compute_75,code=sm_75 \
    -gencode arch=compute_80,code=compute_80	

TARGET = main.exe
CUDA_SOURCE = linear_filter.cu
CPU_SOURCE = main.cpp
CUDA_OBJECT = linear_filter.o
CPU_OBJECT = main.o

# Default target
.PHONY: all
all: build

# Build target
.PHONY: build
build: $(TARGET)

$(TARGET): $(CUDA_OBJECT) $(CPU_OBJECT)
	$(NVCC) $(CXXFLAGS) -o $@ $(CUDA_OBJECT) $(CPU_OBJECT) $(CUDA_LIBS) $(LDFLAGS)

$(CUDA_OBJECT): $(CUDA_SOURCE)
	$(NVCC) $(ARCH_FLAGS) $(CUDA_FLAGS) -c $< -o $@

$(CPU_OBJECT): $(CPU_SOURCE)
	$(GCC) $(CXXFLAGS) -c $< -o $@

# Clean target
.PHONY: clean
clean:
	rm -f $(CUDA_OBJECT) $(CPU_OBJECT) $(TARGET)

# Run target
.PHONY: run
run: build
	./$(TARGET) $(ARGS)

# Phony targets to ensure they are always executed
.PHONY: build clean run
