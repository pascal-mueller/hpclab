CC=mpicc
CFLAGS=-O3

all: poisson

poisson: poisson.c
	$(CC) $(CFLAGS) $^ -lpetsc -o $@

clean:
	rm -f poisson

