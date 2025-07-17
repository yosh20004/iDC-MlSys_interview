# Makefile for test.cpp with OpenBLAS support
CXX = g++
CXXFLAGS = -O3 -Wall -DUSE_BLAS
LDFLAGS = -lopenblas
TARGET = test_openblas
SRC = test.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

clean:
	rm -f $(TARGET)
