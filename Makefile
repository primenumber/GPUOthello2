NVCC=nvcc
NVCCOPT=-g -std=c++17 --compiler-options -march=native -O3 -rdc=true
CXXOPT=-std=c++17 -march=native -O3
OBJS=main.o solver.o thinker.o to_board.o board.o table.o eval.o eval_host.o

.SUFFIXES: .cpp .c .cu .o
.POHNY: clean

all: solver

solver: $(OBJS)
	nvcc -o $@ $(NVCCOPT) $^

test: test.o to_board.o
	g++ -o $@ $(CXXOPT) $^

%.o : %.cu
	$(NVCC) $(NVCCOPT) -c $< -o $@

%.o : %.cpp
	$(CXX) $(CXXOPT) -c $< -o $@

%.cubin : %.cu
	$(NVCC) -cubin $(NVCCOPT) -c $< -o $@

clean:
	-rm *.o
	-rm solver 

solver.o: to_board.hpp alphabeta.cuh solver.cuh types.hpp board.cuh table.cuh node.cuh
thinker.o: to_board.hpp alphabeta.cuh thinker.cuh types.hpp board.cuh table.cuh eval.cuh node.cuh
to_board.o: to_board.hpp types.hpp
main.o: alphabeta.cuh solver.cuh thinker.cuh types.hpp table.cuh eval.cuh to_board.hpp
board.o: board.cuh
table.o: table.cuh board.cuh types.hpp
eval.o: eval.cuh eval_host.hpp types.hpp
eval_host.o: eval_host.hpp types.hpp
test.o: to_board.hpp
