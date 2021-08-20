CXX = g++
LIBS = -I/usr/include/eigen3 -lSDL2
CXXFLAGS = -O2

all: train test draw validate
clean:
	rm -rf train test draw validate
.cpp:
	$(CXX) $(LIBS) $(CXXFLAGS) -o $@ $<
