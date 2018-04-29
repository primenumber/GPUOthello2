NVCC=nvcc
NVCCOPT=-g -std=c++14 --compiler-options "-march=native -mtune=native -Wall -Wextra" -arch=sm_61 -m64 -O3 -lboost_timer -rdc=true
CXXOPT=-std=c++14 -march=native -mtune=native -O3 -Wall -Wextra
OBJS=main.o solver.o to_board.o board.o table.o eval.o eval_host.o expand.o

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

solver.o: to_board.hpp solver.cuh types.hpp board.cuh table.cuh eval.cuh
to_board.o: to_board.hpp types.hpp
main.o: solver.cuh types.hpp table.cuh eval.cuh
board.o: board.cuh
table.o: table.cuh board.cuh types.hpp
eval.o: eval.cuh eval_host.hpp types.hpp
eval_host.o: eval_host.hpp types.hpp
expand.o: eval.cuh board.cuh types.hpp
