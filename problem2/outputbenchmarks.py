import csv
import matplotlib.pyplot as plt
import numpy as np

def getParallelEfficency(results):
    T1 = int(float(results[0][1])) # s

    p_eff = np.zeros(len(results))
    speed_up = np.zeros(len(results))
    
    for (i, result) in enumerate(results):
        T_p = int(float(result[1])) # s
        speed_up[i] = T1/T_p # T1/T_p
        p_eff[i] = speed_up[i]/int(result[0]) #Speedup/p

    return speed_up, p_eff

P = [1,8,16,32]
N = [500, 1000, 2000, 3000]

# Plot
fig, (ax1, ax2) = plt.subplots(1,2)

for (j,n) in  enumerate(N):
    resuts = []
    runtimes = []
    for (i,p) in enumerate(P):
        with open(filename, "rb") as f:
            buf = f.read(12)
            # Data Layout: <int:num_threads>
            #              <double:elapsed_time>
            # data = struct.unpack('=id', buf) FIRST ENTRY IS NUM THREAD SECOND IS DURATION
            results.append(struct.unpack('=id', buf))
       
	print("p=",p,"n=",n)
	print(results[j])
	print("\n\n") 
