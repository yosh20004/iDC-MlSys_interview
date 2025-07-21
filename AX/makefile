CXX = g++
CXXFLAGS = -O0 -Wall
TARGET = test
SRC = test.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

clean:
	rm -f $(TARGET)

