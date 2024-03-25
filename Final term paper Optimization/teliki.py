import numpy as np
from abc import abstractmethod, ABC
import argparse


class Problem(ABC):
    def __init__(self, n):
        self.dimension = n
        self.left = np.zeros(n)
        self.right = np.zeros(n)

        self.bestx = self.getSample()
        self.besty = self.funmin(self.bestx)
    
    def statFunmin(self, x):
        y = self.funmin(x)
        if self.besty > y: 
            self.bestx = x
            self.besty = y 
        return y

    def getDimension(self):
        return self.dimension

    def getSample(self):
        r = np.random.rand(self.dimension)
        x = np.array(self.left) + (np.array(self.right) - np.array(self.left)) * r
        return x.tolist()

    def setLeftMargin(self, x):
        self.left = x

    def setRightMargin(self, x):
        self.right = x

    def getLeftMargin(self):
        return self.left.tolist()

    def getRightMargin(self):
        return self.right.tolist()
 
    @abstractmethod       
    def funmin(self, x):
        pass
    
    @abstractmethod
    def gradient(self, x):
        pass
   
    def grms(self, x):
        g = self.gradient(x)
        s = np.sum(np.square(g))
        return np.sqrt(s / len(x))

class RastriginProblem(Problem):
    def __init__(self):
        super().__init__(2)
        self.left = [-1.0, -1.0]
        self.right = [1.0, 1.0]
    
    def funmin(self, x):
        return x[0] ** 2 + x[1] ** 2 - np.cos(18.0 * x[0]) - np.cos(18.0 * x[1])

    def gradient(self, x):
        g = [0.0] * 2
        g[0] = 2.0 * x[0] + 18.0 * np.sin(18.0 * x[0])
        g[1] = 2.0 * x[1] + 18.0 * np.sin(18.0 * x[1])
        return g

class GradientDescent:
    def __init__(self, p):
        self.myProblem = p
        self.rate = 0.001
        self.xpoint = np.array(self.myProblem.getSample())
        self.ypoint = self.myProblem.funmin(self.xpoint)

    def setRate(self, r):
        if r > 0:
            self.rate = r

    def getRate(self):
        return self.rate

    def updateRate(self):
        pass

    def updatePoint(self):
        g = self.myProblem.gradient(self.xpoint)
        self.xpoint -= self.rate * np.array(g)
        self.ypoint = self.myProblem.statFunmin(self.xpoint)

    def Solve(self):
        k = 0
        while True:
            self.updateRate()
            self.updatePoint()
            k += 1
            if self.myProblem.grms(self.xpoint) < 1e-3:
               break
            print(f"Iteration={k} Value={self.ypoint:.10g}")

class GeneticAlgorithm:
    def __init__(self, p, population_size=50, generations=100, convergence_threshold=1e-6, mutation_rate=0.1, e=1e-6):
        self.myProblem = p
        self.population_size = population_size
        self.generations = generations
        self.convergence_threshold = convergence_threshold
        self.mutation_rate = mutation_rate
        self.e = e

    def run(self):
        population = np.array([self.myProblem.getSample() for _ in range(self.population_size)])
        best_fitness = np.inf

        for generation in range(self.generations):
            fitness = np.apply_along_axis(self.myProblem.funmin, 1, population)
            best_individual_index = np.argmin(fitness)
            current_best_fitness = fitness[best_individual_index]

            maxfitness = fitness.max()
            minfitness = fitness.min()

            if np.abs(maxfitness - minfitness) <= self.convergence_threshold:
                print(f"Convergence reached for Rastrigin. Best fitness: {current_best_fitness}")
                break

            parents = self.tournament_selection(population)
            offspring = self.crossover(parents)
            offspring = self.mutation(offspring)
            population = np.vstack((parents, offspring))

            best_individual = population[best_individual_index]
            population[best_individual_index] = best_individual

            if np.abs(np.max(fitness) - np.min(fitness)) <= self.e:
                print(f"Convergence based on |fmax - fmin| for Rastrigin Best fitness: {current_best_fitness}")
                break

            if current_best_fitness - np.min(fitness) <= self.e:
                 print(f"Convergence based on |fmax - fmin| for Rastrigin. Best fitness: {current_best_fitness}")
                 break

        best_individual_index = np.argmin(fitness)
        best_individual = population[best_individual_index]
        return best_individual

    def tournament_selection(self, population):
        return [max(population[np.random.choice(len(population), size=self.population_size, replace=True)], key=self.myProblem.funmin) for _ in range(self.population_size)]

    def crossover(self, parents):
        offspring = np.empty_like(parents)
        for i in range(0, len(parents), 2):
            parent1, parent2 = parents[i], parents[i + 1]
            crossover_point = np.random.randint(len(parent1))
            offspring[i, :crossover_point] = parent1[:crossover_point]
            offspring[i, crossover_point:] = parent2[crossover_point:]
            offspring[i + 1, :crossover_point] = parent2[:crossover_point]
            offspring[i + 1, crossover_point:] = parent1[crossover_point:]
        return offspring

    def mutation(self, offspring):
        for i in range(len(offspring)):
            mutation_prob = np.random.rand()
            if mutation_prob < self.mutation_rate:
                mutation_point = np.random.randint(len(offspring[i]))
                offspring[i, mutation_point] += np.random.uniform(-0.1, 0.1)
        return offspring

def optimize(problem):
    ga = GeneticAlgorithm(problem, generations=200, e=1e-6)
    best_features = ga.run()
    yval = problem.statFunmin(best_features)
    print(f'[INFO]Best GA Features:{best_features} Objective:{yval}')
    
    grd = GradientDescent(problem)
    grd.Solve()

    # Get best feature set and bound
    print(f'[INFO]Best Bound:{grd.myProblem.bestx}  Value:{grd.myProblem.besty}')
    return grd.myProblem.bestx

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Function selector")
    parser.add_argument("--rst", action='store_true', help="Run the Rastrigin function")
    
    args = parser.parse_args()
    problem = None
    if args.rst:
        problem = RastriginProblem()
    elif args.ros:
        if args.n.isdigit():

            raise ValueError("You need to provide a valid value for N when running Rosenbrock function.")
    else:
        raise ValueError("You did not choose a problem (Rastrigin or Rosenbrock)")

    optimize(problem)

