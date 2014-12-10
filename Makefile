first: haar

#main.o:main.cpp
	#nvcc -G -g -m64 -gencode arch=compute_11,code=sm_11 -c main.cpp -o main.o

haar.o:haar.cu haar.hpp haar_kernels.cu
	nvcc -G -g -m64 -gencode arch=compute_11,code=sm_11 -c haar.cu -o haar.o

image.o:image.cpp image.hpp
	g++ -c image.cpp -o image.o

#haar_kernels.o:haar_kernels.cu haar.hpp haar_kernels.cuh
	#nvcc -m64 -gencode arch=compute_11,code=sm_11 -c haar_kernels.cu -o haar.o

haar:image.o haar.o
	nvcc -G -g -m64 -gencode arch=compute_11,code=sm_11 haar.o image.o -o haar `pkg-config --libs opencv`
	
