#import numpy as np
#from scipy.optimize import minimize


# def camel_function(x):
  # return 4 * x[0]**2 - 2.1 * x[0]**4 + (x[0]**6) / 3 + x[0] * x[1] - 4 * x[1]**2 + 4 * x[1]**4

# def objective_function(x):
   # return camel_function(x)

#initial_guesses = [
   # np.array([0.0, 0.0]),
   # np.array([2.0, 3.0]),
   # np.array([-2.0, -3.0])
#]

#for i, initial_guess in enumerate(initial_guesses):
  #  result = minimize(objective_function, initial_guess, method='Nelder-Mead')
  #  print(f"Αποτέλεσμα {i + 1}:")
  #  print("Ελάχιστο:", result.fun)
  #  print("Τοποθεσία ελάχιστου:", result.x)


import numpy as np
from scipy.optimize import minimize

def camel_function(x):
    return 4 * x[0]**2 - 2.1 * x[0]**4 + (x[0]**6) / 3 + x[0] * x[1] - 4 * x[1]**2 + 4 * x[1]**4

def objective_function(x):
    return camel_function(x)

initial_guess = np.array([0.0, 0.0])

result = minimize(objective_function, initial_guess, method='Nelder-Mead')

print("Ελάχιστο:", result.fun)
print("Τοποθεσία ελάχιστου:", result.x)

