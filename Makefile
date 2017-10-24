COMMONFLAGS = 
COMMONFLAGS += -O3
COMMONFLAGS += -DNDEBUG
#COMMONFLAGS += -g
COMMONFLAGS += -std=c++11

CXXFLAGS = 

CUFLAGS = 
CUFLAGS += -arch=compute_60
CUFLAGS += -D_FORCE_INLINES

LDFLAGS =
LDFLAGS += -lsiloh5
LDFLAGS += -lpthread


all: main.o kernel.o
	nvcc ${COMMONFLAGS} ${LDFLAGS} ./obj/main.o ./obj/kernel.o -o ./cuda_hydro
    
clean:
	rm ./cuda_hydro
	rm ./obj/*

main.o: ./src/main.cpp
	g++ ${COMMONFLAGS} ${CXXFLAGS} ./src/main.cpp -c -o ./obj/main.o

kernel.o: ./src/kernel.cu
	nvcc ${COMMONFLAGS} ${CUFLAGS} ./src/kernel.cu -c -o ./obj/kernel.o
