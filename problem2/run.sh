for p in 1 8 16
do
  for n in 500 1000 2000 3000
  do
    bsub -n $p -R "select[model=XeonGold_6150]" -R "rusage[mem=2048]" mpirun ./poisson -mat_type mpiaij -n $n 
  done
done 


for n in 500 1000 2000 3000
do
    bsub -n 32 -R "select[model=XeonGold_6150]" mpirun ./poisson -mat_type mpiaij -n $n 
done
