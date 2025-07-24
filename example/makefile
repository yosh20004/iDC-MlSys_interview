CXX = g++-11
NVCC = nvcc

CXXFLAGS = -O3 -Wall -DUSE_BLAS -mfma -mavx2 -fopenmp -std=c++17 -IAX
NVCCFLAGS = -O3 -std=c++17 -ccbin $(CXX) -IAX

LDFLAGS = -L/usr/local/cuda/lib64 -lcusparse -lcudart -fopenmp

TARGET = example.exe

SRC_CPP = gcn.cpp
SRC_CU = AX/cuda_gemm.cu

OBJ_CPP = $(SRC_CPP:.cpp=.o)
OBJ_CU = $(SRC_CU:.cu=.o)
OBJECTS = $(OBJ_CPP) $(OBJ_CU)

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJECTS) AX/*.o