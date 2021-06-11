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

for (j,n) enumerate(N):
    resuts = []
    runtimes = []
    for (i,p) in enumerate(P):
        with open(filename, "rb") as f:
            buf = f.read(12)
            # Data Layout: <int:num_threads>
            #              <double:elapsed_time>
            results.append(struct.unpack('=id', buf))
        
        runtimes.append(results[i][1])
    speed_up, p_eff = getParallelEfficency(runtimes, P)

    # Plot runtimes
    runtimes = results
    ax1.plot(P, runtimes)
    ax1.invert_yaxis()
    ax1.set(xlabel='Number of threads [-]', ylabel='Runtime [s]', title='Runtimes')

    # Plotspeed_up 
    ax2.plot(P, speed_up)
    ax2.set(xlabel='Number of threads [-]', ylabel='Strong Speedup [-]', title='Strong scaling')

plt.tight_layout()
    
fig.savefig(f"strong_scaling.png")
plt.show()
        


