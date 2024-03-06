import numpy as np
import scipy.optimize as opt

def camel_function(x):
    return 4 * x[0]**2 - 2.1 * x[0]**4 + (x[0]**6) / 3 + x[0] * x[1] - 4 * x[1]**2 + 4 * x[1]**4

def objective_function(x):
    return camel_function(x)


num_iterations = 30
minima = []
found_global_min = 0

for i in range(num_iterations):
    initial_guess = np.random.uniform(-5, 5, size=(2,))
    result = opt.dual_annealing(objective_function, bounds=((-5, 5), (-5, 5)), x0=initial_guess)
    minima.append(result.fun)
    
    if np.isclose(result.fun, -1.0316, atol=1e-2):
        found_global_min = 1

print("Ελάχιστες τιμές που βρέθηκαν:", minima)
print("Αριθμός ευρετηρίων ελαχίστου:", found_global_min)
