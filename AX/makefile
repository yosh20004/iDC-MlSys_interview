CXX = g++
CXXFLAGS = -O0 -Wall -I/usr/include/mkl
TARGET = test
SRC = test.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

LDFLAGS  = -L/usr/lib/x86_64-linux-gnu -lmkl_rt -lpthread -lm -ldl

clean:
	rm -f $(TARGET)

