IDIR=include
CC=clang++
CFLAGS=-I$(IDIR) -O3
CVER=--std=c++17
EFLAGS=--cuda-gpu-arch=sm_61

ODIR=obj

LIBS=-lm -lrt -ldl -lcudart
ELIBS=-ltbb -pthread

_DEPS = gpu.hpp
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

_OBJ = reduce_with_index.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

$(ODIR)/%.o: benchmarking/%.cu $(DEPS)
	mkdir -p obj
	$(CC) -c -o $@ $< $(CFLAGS) $(CVER) $(EFLAGS)

reduce_with_index: $(ODIR)/reduce_with_index.o
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS) $(ELIBS) $(CVER)


.PHONY: bench
bench: reduce_with_index

nvcc: CC=nvcc
nvcc: CVER=--std=c++14
nvcc: ELIBS=
nvcc: EFLAGS=
nvcc: bench

.PHONY: clean

clean:
	rm -rf $(ODIR) *~ $(INCDIR)/*~
