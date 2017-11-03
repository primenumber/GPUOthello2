NVCC=nvcc
NVCCOPT=-std=c++14 --compiler-options -march=native -arch=sm_61 -m64 -O3
CXXOPT=-std=c++14 -march=native -O3
OBJS=main.o solver.o to_board.o

.SUFFIXES: .cpp .c .cu .o
.POHNY: clean

all: solver

solver: $(OBJS)
	nvcc -o $@ $(NVCCOPT) $^

%.o : %.cu
	$(NVCC) $(NVCCOPT) -c $< -o $@

%.o : %.cpp
	gcc-6 $(CXXOPT) -c $< -o $@

%.cubin : %.cu
	$(NVCC) -cubin $(NVCCOPT) -c $< -o $@

clean:
	-rm *.o
	-rm solver 

solver.o: to_board.hpp solver.cuh types.hpp
to_board.o: to_board.hpp types.hpp
main.o: solver.cuh types.hpp
