#include "petsc.h"
#include "petscksp.h"
#include <stdlib.h>

void saveBenchmark(double elapsed_time, int num_threads, int N) {
  int n = 100;
  // Create filename
  char filename[n];
  snprintf(filename, n, "benchmarks/benchmark_%d_%d.bin", N, num_threads);

  FILE* pFile = fopen(filename, "wb");

  // Write number of threads
  fwrite(&num_threads, sizeof(char), sizeof(int), pFile);
  // Write times
  fwrite(&elapsed_time, sizeof(char), sizeof(double), pFile);
  fclose(pFile);
}


Mat Laplacian2d(int n){
    Mat A;
    int r , rowStart, rowEnd, i, j;
    double h, overHsquared;

    h = 1.0 / (n+1);
    overHsquared = 1.0 / (h * h);
    MatCreate(PETSC_COMM_WORLD, &A);
    MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n*n, n*n);
    MatSetFromOptions(A);
    
    //set up the matrix data structure for efficient use => hopefully gets rid of segfault
    MatSetUp(A);

    //get region of matrix that is owned by this process
    MatGetOwnershipRange(A, &rowStart, &rowEnd);


    //Assemble Matrix for inner vertices only. With stencil:
                                //     -1
                                // -1   4   -1
                                //     -1
    for (r = rowStart; r < rowEnd; r++){

        //get coordinates of vertex in mesh
        i = r % n;
        j = r / n;

        if (j - 1 > 0){
            MatSetValue(A, r, r - n, overHsquared, INSERT_VALUES);
        }
        if (i - 1 > 0){
            MatSetValue(A, r, r - 1, overHsquared, INSERT_VALUES);
        }

        MatSetValue(A, r, r, -4 * overHsquared, INSERT_VALUES);

        if (i + 1 < n - 1) {
            MatSetValue(A, r, r + 1, overHsquared, INSERT_VALUES); 
        }
        if (j + 1 < n - 1) {
            MatSetValue(A, r, r + n, overHsquared, INSERT_VALUES);
        }
    }
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
    
    return A;
}


Vec FormVecFromConstant2d(int n, double the_constant){
    Vec V;
    int r, rowStart, rowEnd, i, j;
    double h;

    h = 1.0 / (n + 1);
    VecCreate(PETSC_COMM_WORLD, &V);
    VecSetSizes(V, PETSC_DECIDE, n*n);
    VecSetFromOptions(V);
    VecGetOwnershipRange(V, &rowStart, &rowEnd);

    for (r = rowStart; r < rowEnd; r++){
        VecSetValue(V, r, the_constant, INSERT_VALUES);
    }
    VecAssemblyBegin(V);
    VecAssemblyEnd(V);

    return V;
}


int main( int argc, char *argv[] ){
    
    KSP solver;
    Mat A;
    Vec b, x;
    int n, its;

    PetscInitialize(&argc, &argv, 0, 0);
    //Get number of threads
    int size;
    MPI_Comm_size(PETSC_COMM_WORLD, &size);
    printf("\n\n%d", size);
    
    // read grid size option from cmdl
    PetscOptionsGetInt(PETSC_NULL, NULL, "-n", &n, 0);

    //Assemble Matrix and vectors
    A = Laplacian2d(n);
    b = FormVecFromConstant2d(n, 20.0);
    VecDuplicate(b, &x);

    //Set up KSP solver
    KSPCreate(PETSC_COMM_WORLD, &solver);
    KSPSetOperators(solver, A, A);
    KSPSetFromOptions(solver);

    //Solve
    double time_start = MPI_Wtime();
    KSPSolve(solver, b, x);
    double time_end = MPI_Wtime();

    //Get info about solution process
    KSPGetIterationNumber(solver, &its);
    PetscPrintf(PETSC_COMM_WORLD, "Solution took %d iterations\n)");
    VecView(x, PETSC_VIEWER_STDOUT_WORLD);
    
    //write vector to file
    PetscViewer viewer;
    
    
    // Create filename
    char filename[100];
    snprintf(filename, 100, "data/vector_%d_%d.txt", n, size);
    
    PetscViewerASCIIOpen(PETSC_COMM_WORLD, filename, &viewer);
    PetscViewerPushFormat(viewer, PETSC_VIEWER_DEFAULT);
    VecView(x,viewer);
    PetscViewerPopFormat(viewer);
    PetscViewerDestroy(&viewer);

    //Free memory
    MatDestroy(&A);
    VecDestroy(&b);
    VecDestroy(&x);
    KSPDestroy(&solver);
    

    
    
    PetscFinalize( );

    //Save benchmark
    saveBenchmark(time_end - time_start, size, n);

    return 0;
}

