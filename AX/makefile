CXX = clang++
CXXFLAGS = -O3 -Wall -I/usr/include/mkl -fopenmp
TARGET = test
SRC = test.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

LDFLAGS  = -L/usr/lib/x86_64-linux-gnu -lmkl_rt -lpthread -lm -ldl -fopenmp -mfma -mavx2

clean:
	rm -f $(TARGET)

