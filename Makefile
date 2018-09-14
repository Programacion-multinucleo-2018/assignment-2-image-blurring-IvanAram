C_C = g++
CUDA_C = nvcc

CFLAGS = -std=c++11 -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs
LDFLAGS = -lpthread

EXE1 = bin/blur_cpu
EXE2 = bin/blur_threads
EXE3 = bin/blur_cuda

PROG1 = blur_cpu.cpp
PROG2 = blur_threads.cpp
PROG3 = blur_cuda.cu

all:
	$(C_C) -o $(EXE1) $(PROG1) $(CFLAGS)
	$(C_C) -o $(EXE2) $(PROG2) $(CFLAGS) $(LDFLAGS)
	$(CUDA_C) -o $(EXE3) $(PROG3) $(CFLAGS)

rebuild: clean all

clean:
	rm -f ./bin/*
