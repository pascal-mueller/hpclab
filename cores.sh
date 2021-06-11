#!/bin/bash

echo "How many cores?"
read p

export OMP_NUM_THREADS=$p
export OPENBLAS_NUM_THREADS=$p 
export MKL_NUM_THREADS=$p
export VECLIB_MAXIMUM_THREADS=$p
export NUMEXPR_NUM_THREADS=$p

echo "Set:"

echo "$OMP_NUM_THREADS"
echo "$OPENBLAS_NUM_THREADS"
echo "$MKL_NUM_THREADS"
echo "$VECLIB_MAXIMUM_THREADS"
echo "$NUMEXPR_NUM_THREADS"
    



