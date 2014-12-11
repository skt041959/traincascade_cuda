#first: haar
first:adaboost

#main.o:main.cpp
	#nvcc -G -g -m64 -gencode arch=compute_11,code=sm_11 -c main.cpp -o main.o

haar.o:haar.cu haar.hpp haar_kernels.cu
	nvcc -G -g -m64 -gencode arch=compute_11,code=sm_11 -c haar.cu -o haar.o

image.o:image.cpp image.hpp
	g++ -g -c image.cpp -o image.o

#haar_kernels.o:haar_kernels.cu haar.hpp haar_kernels.cuh
	#nvcc -m64 -gencode arch=compute_11,code=sm_11 -c haar_kernels.cu -o haar.o

adaboost.o:adaboost.cu adaboost.hpp
	nvcc -G -g -m64 -gencode arch=compute_11,code=sm_11 -c adaboost.cu -o adaboost.o

main.o:main.cpp
	g++ -g -c main.cpp -o main.o

adaboost:adaboost.o haar.o image.o main.o
	nvcc -G -g -m64 -gencode arch=compute_11,code=sm_11 haar.o image.o adaboost.o main.o -o adaboost `pkg-config --libs opencv`
