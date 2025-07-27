CXX = g++-11
NVCC = nvcc

CXXFLAGS = -O3 -Wall -mfma -mavx2 -fopenmp -std=c++17
NVCCFLAGS = -O3 -std=c++17 -ccbin $(CXX) --extended-lambda -lcusparse

LDFLAGS = -L/usr/local/cuda/lib64 -lcudart

TARGET = ./test

SRC_CPP = test.cpp
SRC_CU = kernel.cu

OBJ_CPP = $(SRC_CPP:.cpp=.o)
OBJ_CU = $(SRC_CU:.cu=.o)
OBJECTS = $(OBJ_CPP) $(OBJ_CU)

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(NVCC) $(NVCCFLAGS) $(OBJECTS) -o $@ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJECTS) ./*.o