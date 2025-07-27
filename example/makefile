CXX = g++-11
NVCC = nvcc

INC_DIR = -I. -I.. -I$(shell pwd) -I$(shell pwd)/.. -I/usr/local/cuda/include

CXXFLAGS = -O3 -Wall -DUSE_BLAS -mfma -mavx2 -fopenmp -std=c++17 -IAX $(INC_DIR)
NVCCFLAGS = -O3 -std=c++17 -ccbin $(CXX) -IAX $(INC_DIR)

LDFLAGS = -L/usr/local/cuda/lib64 -lcusparse -lcudart -lcublas -fopenmp

TARGET = example.exe

SRC_CPP = gcn.cpp
SRC_CU = cpu/AX/cuda_gemm.cu cuda/AX/gemm.cu

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
	rm -f $(TARGET) $(OBJECTS) cpu/AX/*.o
	rm -f $(TARGET) $(OBJECTS) cuda/AX/*.o
	rm -f $(TARGET) $(OBJECTS) cpu/XW/*.o
	rm -f $(TARGET) $(OBJECTS) cuda/XW/*.o