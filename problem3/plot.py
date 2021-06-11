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

with open('results.csv') as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    results = list(reader)

# Turn results list into np array
results = np.array(results)


# Get num threads
P = results[:,0]

# Get runtimes
times = results[:,1]

for (i, time) in enumerate(times):
    times[i] = int(float(times[i])) / 1000

# Get parallel efficency for strong scaling
speed_up, p_eff = getParallelEfficency(results)

# Plot
fig, (ax1, ax2) = plt.subplots(1,2)

# Plot runtimes
ax1.plot(P, times)
ax1.invert_yaxis()
ax1.set(xlabel='Number of threads [-]', ylabel='Runtime [s]', title='Runtimes')

# Plotspeed_up 
ax2.plot(P, speed_up)
ax2.set(xlabel='Number of threads [-]', ylabel='Strong Speedup [-]', title='Strong scaling')

plt.tight_layout()

fig.savefig("results.png")
plt.show()
