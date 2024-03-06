import numpy as np
import scipy.optimize as opt


def objective_function(x):
    return 0.5 * np.sum(x**(4*np.arange(1,5)) - 16 * x**(2*np.arange(1,5)) + 5 * x)

minima_annealing = []


for _ in range(30):
    initial_guess = np.random.uniform(-5, 5, size=(4,))
    result = opt.dual_annealing(objective_function, bounds=[(-5,5)]*4, x0=initial_guess)
    minima_annealing.append(result.fun)
    

count_global_min = minima_annealing.count(-156.664663)

print("Ελάχιστες τιμές που βρέθηκαν:", minima_annealing)
print("Πόσες φορές βρέθηκε το ολικό ελάχιστο:", count_global_min)
