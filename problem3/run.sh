#!/bin/bash

for p in 1 4 8 12 16 24
do
  echo "Running script with $p threads"
  python toy.py $p
done
