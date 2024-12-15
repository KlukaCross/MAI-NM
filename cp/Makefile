CPP = g++

all: build

build:
	$(CPP) lup_decompose_sequential.cpp -o s.out
	$(CPP) -fopenmp lup_decompose_parallel.cpp -o p.out

clean:
	rm -rf *.o *.out
