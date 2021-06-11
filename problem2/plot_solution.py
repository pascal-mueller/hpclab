import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
import glob, os

for filename in glob.glob("data/*.txt"):
    tmp = filename.replace(".txt", "")
    tmp = tmp.replace("data/vector", "")
    tmp = tmp.split("_")
    p = int(float(tmp[2]))
    N = int(float(tmp[1]))
    

    # Read solution data
    u = np.genfromtxt(filename, dtype=np.float64, delimiter='\n', skip_header=3)

    # Remove the "Process [x]" entries
    u = u[np.logical_not(np.isnan(u))] 

    z = np.reshape(u, (-1, N))
    
    # Vanishing dirichlet BC
    K = np.zeros( (N+2,N+2) )
    K[1:-1,1:-1] = z


    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})


    x = np.linspace(0,1,N)
    y = np.linspace(0,1,N)
    x,y = np.meshgrid(x,y)

    surf = ax.plot_surface(x,y,z)

    plt.show()
