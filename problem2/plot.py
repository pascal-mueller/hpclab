import os
import csv
import matplotlib.pyplot as plt
import numpy as np
import struct

def getParallelEfficency(results):
    T1 = int(float(results[0][1])) # s

    p_eff = np.zeros(len(results))
    speed_up = np.zeros(len(results))
    
    for (i, result) in enumerate(results):
        T_p = float(result[1]) # s
        #print(f"{T_p} for {result}")
        speed_up[i] = T1/T_p # T1/T_p
        p_eff[i] = speed_up[i]/int(result[0]) #Speedup/p

    return speed_up, p_eff

P = [1,8,16,32]
N = [500, 1000, 2000, 3000]

# Plot
fig, (ax1, ax2) = plt.subplots(1,2)

for (j,n) in enumerate(N):
    results = []
    runtimes = []
    PP = [] 
    for (i,p) in enumerate(P):
        filename = f"benchmarks/benchmark_{n}_{p}.bin"
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                buf = f.read(12)
                # Data Layout: <int:num_threads>
                #              <double:elapsed_time>
                data = struct.unpack('=id', buf)
                data = np.array([data[0], data[1]])
                results.append(data)
                runtimes.append(data[1])
                PP.append(p)
        else:
            print("file missing:", filename)

    speed_up, p_eff = getParallelEfficency(results)


    # Plot runtimes
    ax1.plot(PP, runtimes, '-o', label=f"N={n}")
    ax1.set_xticks(PP)
    ax1.invert_yaxis()
    ax1.set(xlabel='Number of threads [-]', ylabel='Runtime [s]', title='Runtimes')

    # Plotspeed_up 
    ax2.plot(PP, speed_up, '-o', label=f"N={n}")
    ax1.set_xticks(PP)
    ax2.set(xlabel='Number of threads [-]', ylabel='Strong Speedup [-]', title='Strong scaling')

ax1.legend()
ax2.legend()
plt.tight_layout()
    
fig.savefig(f"strong_scaling.png")
plt.show()
        


