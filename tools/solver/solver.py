import inspect
from scipy.sparse import linalg
import numpy as np

array_it, array_frame, it_count = np.array([]), np.array([]), 0

def gmres(op_discrete,rhs,tol_solver=1e-6):

    def iteration_counter(x):
            global array_it, array_frame, it_count
            it_count += 1
            frame = inspect.currentframe().f_back
            array_it = np.append(array_it, it_count)
            array_frame = np.append(array_frame, frame.f_locals["resid"])    
    x, info = linalg.gmres(op_discrete, rhs, callback=iteration_counter, tol=tol_solver, maxiter=10000, restart = 20000)

    return x, info, it_count