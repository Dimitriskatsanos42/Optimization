import numpy as np
import random
import math


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

class NewFunctionProblem(Problem):
    def __init__(self):
        super().__init__(2)
        self.left = [-5.0, -5.0]
        self.right = [5.0, 5.0]

    def funmin(self, x):
        return 4 * x[0]**2 - 2.1 * x[0]**4 + (x[0]**6) / 3 + x[0] * x[1] - 4 * x[1]**2 + 4 * x[1]**4

    def gradient(self, x):
        g = [0] * 2
        g[0] = 8 * x[0] - 8.4 * x[0]**3 + (x[0]**5) - 4 * x[1]
        g[1] = x[0] - 8 * x[1] + 16 * x[1]**3
        return g


class SimanMethod:
    def __init__(self, p):
        self.myProblem = p
        self.T0 = 100000.0
        self.xpoint = self.myProblem.getSample()
        self.ypoint = self.myProblem.funmin(self.xpoint)
        self.neps = 100
        self.eps = 1e-5

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
        k = 1
        while True:
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
            self.T0 = self.T0 * alpha**k
            k += 1
            if self.T0 <= self.eps:
               print(f'Iteration: {k} Temperature: {self.T0} Value: {self.ypoint}')
               break

    def getX(self):
        return self.xpoint

if __name__ == '__main__':
    num_executions = 30
    results = []

    for _ in range(num_executions):
        problem = NewFunctionProblem()
        sa = SimanMethod(problem)
        sa.Solve()
        results.append(sa.getX())

    for i, x_opt in enumerate(results):
        print(f'Execution {i + 1}:')
        print('Optimal Solution:', x_opt)
        print('Optimal Value:', problem.funmin(x_opt))
