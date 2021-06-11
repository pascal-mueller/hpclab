from threadpoolctl import threadpool_limits
import time
import numpy as np
import sys

p = sys.argv[1]

with threadpool_limits(limits=int(p), user_api='blas'):
    # In this block, calls to blas implementation (like openblas or MKL)
    # will be limited to use only one thread. They can thus be used jointly
    # with thread-parallelism.
    a = np.random.randn(10000, 10000)
    start = time.time()
    a_squared = a @ a
    stop = time.time()

duration = (stop - start)*1000

with open("results.csv", "a") as f:
    result = f"{p},{duration}\n"
    f.write(result)

print("p=", p, " => ", (stop-start)*1000)
