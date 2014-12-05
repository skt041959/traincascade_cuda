first: haar

main.o:main.cpp
	nvcc -m64 -gencode arch=compute_11,code=sm_11 -c main.cpp -o main.o

haar.o:haar.cu haar.hpp
	nvcc -m64 -gencode arch=compute_11,code=sm_11 -c haar.cu -o haar.o

haar:haar.o main.o
	nvcc -m64 -gencode arch=compute_11,code=sm_11 haar.o main.o -o haar `pkg-config --libs opencv`
	
