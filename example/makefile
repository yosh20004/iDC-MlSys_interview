CXX = g++
CXXFLAGS = -O3 -Wall -DUSE_BLAS -mfma -mavx2 -fopenmp
LDFLAGS = -lopenblas
TARGET = example.exe
SRC = gcn.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

clean:
	rm -f $(TARGET)

