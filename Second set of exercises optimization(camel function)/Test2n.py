import numpy as np
import random
import math
from scipy.optimize import minimize

class Problem:
    def __init__(self, n):
        self.dimension = n
        self.left = [0] * n
        self.right = [0] * n

    def getDimension(self):
        return self.dimension

    def getSample(self):
        x = [0] * self.dimension
        for i in range(self.dimension):
            r = random.uniform(0, 1)
            if r < 0:
                r = -r
            x[i] = self.left[i] + (self.right[i] - self.left[i]) * r
        return x

    def setLeftMargin(self, x):
        self.left = x

    def setRightMargin(self, x):
        self.right = x

    def getLeftMargin(self):
        return self.left

    def getRightMargin(self):
        return self.right

    def funmin(self, x):
        pass

    def gradient(self, x):
        pass

    def grms(self, x):
        g = self.gradient(x)
        s = sum(gi ** 2 for gi in g)
        return math.sqrt(s / len(x))


class RastriginProblem(Problem):
    
    def __init__(self):
        super().__init__(4)  # allazo tin diastasi se 4
        self.left = [-5.0, -5.0, -5.0, -5.0]
        self.right = [5.0, 5.0, 5.0, 5.0]

    def funmin(self, x):
        n = len(x)
        sum1 = sum(x[i]**4 - 16 * x[i]**2 + 5 * x[i] for i in range(n))
        return 1/2 * sum1

    def gradient(self, x):
        g = [0] * 2
        g[0] = 2.0 * x[0] + 18.0 * math.sin(18.0 * x[0])
        g[1] = 2.0 * x[1] + 18.0 * math.sin(18.0 * x[1])
        return g

class Simplex:
    def simplex_algorithm_for_RastriginProblem(self, initial_guess):
        result = minimize(RastriginProblem().funmin, initial_guess, method='Nelder-Mead')
        return result

class SimanMethod:
    def __init__(self, p):
        self.myProblem = p
        self.T0 = 1000
        self.xpoint = self.myProblem.getSample()
        self.ypoint = self.myProblem.funmin(self.xpoint)
        self.neps = 100
        self.eps = 1e-5
        self.k = 1  

    def setT0(self, t):
        if t > 0:
            self.T0 = t

    def getT0(self):
        return self.T0

    def setNeps(self, n):
        if n > 0:
            self.neps = n

    def getNeps(self):
        return self.neps

    def setEpsilon(self, e):
        if e > 0:
            self.eps = e

    def getEpsilon(self):
        return self.eps

    def Solve(self):
        max_iterations = 30
        while self.k <= max_iterations:  
            for i in range(self.neps):
                y = self.myProblem.getSample()
                fy = self.myProblem.funmin(y)
                if fy < self.ypoint:
                    self.xpoint = y
                    self.ypoint = fy
                else:
                    r = random.random()
                    ratio = math.exp(-(fy - self.ypoint) / self.T0)
                    xmin = ratio if ratio < 1 else 1
                    if r < xmin:
                        self.xpoint = y
                        self.ypoint = fy
            alpha = 0.8
            self.T0 = self.T0 * alpha**self.k  
            self.k += 1
            print(f'Iteration: {self.k} Temperature: {self.T0} Value: {self.ypoint}')  
            
            
    def getX(self):
        return self.xpoint

if __name__ == '__main__':
    num_executions = 30
    results_simplex = []
    results_simulated_annealing = []
    
    for _ in range(num_executions):


        result_simplex = Simplex().simplex_algorithm_for_RastriginProblem([random.uniform(-5, 5)] * 4)
        results_simplex.append(result_simplex)
        #result_simplex = Simplex().simplex_algorithm_for_RastriginProblem([random.uniform(-5, 5), random.uniform(-5, 5)])

        problem = RastriginProblem() #efarmozo tin methodo Simulated Annealing
        sa = SimanMethod(problem)
        sa.Solve()
        results_simulated_annealing.append(sa.getX())
    

    for i, x_opt in enumerate(results_simplex):
        print(f'Simplex Execution {i + 1}:')
        print('Optimal Solution:', x_opt.x)
        print('Optimal Value:', x_opt.fun)

    for i, x_opt in enumerate(results_simulated_annealing):
        print(f'Simulated Annealing Execution {i + 1}:')
        print('Optimal Solution:', x_opt)
        print('Optimal Value:', problem.funmin(x_opt))
