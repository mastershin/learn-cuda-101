# Define variables
NVCC = nvcc
GCC = g++
CXXFLAGS = -std=c++17 -g
# CUDA_FLAGS = -arch=sm_75 -Xcompiler -fPIC
CUDA_FLAGS = -Xcompiler -fPIC
# CUDA architecture flags
ARCH_FLAGS = \
    -gencode arch=compute_60,code=sm_60 \
    -gencode arch=compute_70,code=sm_70 \
    -gencode arch=compute_75,code=sm_75 \
    -gencode arch=compute_80,code=sm_80 \
    -gencode arch=compute_80,code=compute_80	

TARGET = reduce.ex
CUDA_SOURCE = reduce.cu
CPU_SOURCE = main.cpp
CUDA_OBJECT = reduce.o
CPU_OBJECT = main.o

# Default target
.PHONY: all
all: build

# Build target
.PHONY: build
build: $(TARGET)

$(TARGET): $(CUDA_OBJECT) $(CPU_OBJECT)
	$(NVCC) -o $@ $(CUDA_OBJECT) $(CPU_OBJECT) -lcuda

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