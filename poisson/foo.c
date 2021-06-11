// module load new gcc open_mpi petsc

#include <math.h> 
#include <petsc.h>
#include <petscksp.h>
#include <petscvec.h>

// Solves Ax=b where A is the laplacian operator sparse matrix
// This eq represents the discretized version of the poission eq -delta(x) = f

Mat FormLaplacian2d(int n)
{
    Mat A;
    int r, rowStart, rowEnd, i, j;
    double h, oneByh2;

    h = 1.0 / (n + 1); oneByh2 = 1.0 / (h * h);
    
    // Create shared MPI matrix with specific sizes
    MatCreate(PETSC_COMM_WORLD, &A);
    MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n*n, n*n);
    MatSetFromOptions(A);
    // Find which part of the matrix belongs to this rank
    MatGetOwnershipRange(A, &rowStart, &rowEnd);
    
    MatSetUp(A);
    // Insert values into our part of the matrix
    for (r=rowStart; r<rowEnd; r++) {
        i = r % n; j = r / n;

        // Setup laplacian matrix, banded diagonal
        if (j - 1 > 0)
            MatSetValue(A, r, r - n, oneByh2, INSERT_VALUES);
        if (i - 1 > 0)
            MatSetValue(A, r, r - 1, oneByh2, INSERT_VALUES);

        MatSetValue(A, r, r, -4 * oneByh2, INSERT_VALUES);

        if (i + 1 < n - 1)
            MatSetValue(A, r, r + 1, oneByh2, INSERT_VALUES);
        if (j + 1 < n - 1)
        
        MatSetValue(A, r, r + n, oneByh2, INSERT_VALUES);
    }

    // Send other ranks our part of the matrix
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    // Wait for the IBcast to finish
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
    
    return A;
}

Vec FormVecFromFunction2d(int n, double (*f)(double, double))
{
    Vec V;
    int r, rowStart, rowEnd, i, j;
    double h = 1.0 / (n + 1);
    
    // Similar to the matrix but for the rhs of the equation Ax=b
    VecCreate(PETSC_COMM_WORLD, &V);
    VecSetSizes(V, PETSC_DECIDE, n*n);
    VecSetFromOptions(V);
    VecGetOwnershipRange(V, &rowStart, &rowEnd);

    VecSetUp(V);
    // This is a simple but inefficient way to set the vector
    for (r=rowStart; r<rowEnd; r++) {
        i = (r % n) + 1;
        j = (r / n) + 1;

        // Use provided function
        VecSetValue(V, r, (*f)(i * h, j * h), INSERT_VALUES);
    }

    // Communicate with other ranks
    VecAssemblyBegin(V);
    VecAssemblyEnd(V);

    return V;
}

// the rhs function
double func(double x, double y) {
    return sin(x*M_PI)*sin(y*M_PI); 
}

int main(int argc, char* argv[])
{
    // the LSE solver, krylov space
    KSP sles;
    Mat A;
    Vec b, x;
    int its, n;

    PetscInitialize(&argc, &argv, 0, 0);
    n = 10; // Get the mesh size. Use 10 by default

    // read options from commandline args
    PetscOptionsGetInt(PETSC_NULL, NULL, "-n", &n, 0);

    // Create matrix and vector using the functions above
    A = FormLaplacian2d(n);
    b = FormVecFromFunction2d(n, func);
    // Copy b into the initial guess for x
    VecDuplicate(b, &x);

    // Create the solver and initialize it, solver is chosen by petsc?
    KSPCreate(PETSC_COMM_WORLD, &sles);
    KSPSetOperators(sles, A, A);
    KSPSetFromOptions(sles);

    // Perform the actual solving
    KSPSolve(sles, b, x);

    // Read output and print it
    KSPGetIterationNumber(sles, &its);
    PetscPrintf(PETSC_COMM_WORLD, "Solution in %d iterations is:\n", );
    VecView(x, PETSC_VIEWER_STDOUT_WORLD);

    // Delete data
    MatDestroy(&A); VecDestroy(&b); VecDestroy(&x);
    KSPDestroy(&sles);

    PetscFinalize();
    return 0;
}

