import numpy as np
from scipy.optimize import minimize


def objective_function(x):
    return 0.5 * np.sum(x**(4*np.arange(1,5)) - 16 * x**(2*np.arange(1,5)) + 5 * x)

minima_simplex = []


for _ in range(30):
    initial_guess = np.random.uniform(-5, 5, size=(4,))
    result = minimize(objective_function, initial_guess, method='Nelder-Mead')
    minima_simplex.append(result.fun)
    

count_global_min = minima_simplex.count(-156.664663)

print("Ελάχιστες τιμές που βρέθηκαν:", minima_simplex)
print("Πόσες φορές βρέθηκε το ολικό ελάχιστο:", count_global_min)
