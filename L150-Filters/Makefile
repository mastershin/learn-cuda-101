# IDIR=./
CXX = nvcc

CXXFLAGS += $(shell pkg-config --cflags --libs opencv4)
LDFLAGS += $(shell pkg-config --libs --static opencv)

all: clean build

build: 
	$(CXX) linear_filter.cu --std c++17 `pkg-config opencv --cflags --libs` -o linear_filter.exe -Wno-deprecated-gpu-targets $(CXXFLAGS) -I/usr/local/cuda/include -lcuda

run:
	./linear_filter.exe $(ARGS)

clean:
	rm -f linear_filter.exe
