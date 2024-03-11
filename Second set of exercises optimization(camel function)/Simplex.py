import numpy as np

class Simplex:
    def __init__(self, c, A, b):
        self.c = c
        self.A = A
        self.b = b
        self.num_constraints, self.num_variables = A.shape

    def initialize(self):
        self.basic_variables = np.arange(self.num_variables, self.num_variables + self.num_constraints)
        self.non_basic_variables = np.arange(self.num_variables)
        self.tableau = self.create_initial_tableau()

    def create_initial_tableau(self):
        tableau = np.zeros((self.num_constraints + 1, self.num_variables + self.num_constraints + 1))
        tableau[0, :self.num_variables] = -self.c
        tableau[1:, :self.num_variables] = self.A
        tableau[1:, self.num_variables:self.num_variables + self.num_constraints] = np.eye(self.num_constraints)
        tableau[1:, -1] = self.b
        return tableau

    def find_pivot(self):
        col = np.argmax(self.tableau[0, :-1] > 0)
        if self.tableau[0, col] <= 0:
            return None  
        ratios = self.tableau[1:, -1] / self.tableau[1:, col]
        min_ratio = np.min(ratios)
        row = np.argmin(ratios)
        return row, col

    def pivot(self, row, col):
        self.basic_variables[row] = col
        pivot_value = self.tableau[row, col]
        self.tableau[row, :] /= pivot_value
        for i in range(self.num_constraints + 1):
            if i != row:
                factor = self.tableau[i, col]
                self.tableau[i, :] -= factor * self.tableau[row, :]

    def is_optimal(self):
        return np.all(self.tableau[0, :-1] <= 0)

    def optimize(self):
        self.initialize()
        while not self.is_optimal():
            pivot = self.find_pivot()
            if pivot is None:
                return None  
            self.pivot(*pivot)
        optimal_solution = np.zeros(self.num_variables)
        for i in range(self.num_constraints):
            if self.basic_variables[i] < self.num_variables:
                optimal_solution[self.basic_variables[i]] = self.tableau[i + 1, -1]
        return optimal_solution

def camel_function(x):
    return 4 * x[0]**2 - 2.1 * x[0]**4 + (x[0]**6) / 3 + x[0] * x[1] - 4 * x[1]**2 + 4 * x[1]**4

def run_simplex_on_camel_function():
    for i in range(30):
        initial_guess = np.random.uniform(-5, 5, size=(2,))
        c = np.array([0, 0])
        A = np.array([[0, 0]])
        b = np.array([0])
        simplex = Simplex(c, A, b)
        optimal_solution = simplex.optimize()
        if optimal_solution is not None:
            minimum = camel_function(optimal_solution)
            print(f"Iteration {i + 1}:")
            print("Initial Guess:", initial_guess)
            print("Minimum:", minimum)
            print("Location of Minimum:", optimal_solution)
            print()

run_simplex_on_camel_function()

