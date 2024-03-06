import math


def f(x):
    return x * (x - 1) + x**2 * (x - 3)


def golden_section_search(f, a, b, tol=1e-6):
    phi = (1 + math.sqrt(5)) / 2  

    x1 = a + (b - a) / phi
    x2 = b - (b - a) / phi
    f1 = f(x1)
    f2 = f(x2)

    steps = []

    while (b - a) > tol:
        steps.append((a, b))
        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + (b - a) / phi
            f1 = f(x1)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = b - (b - a) / phi
            f2 = f(x2)
            print(f'[INFO]{a=}\t{b=}')

    x_optimal = (a + b) / 2
    return x_optimal, f(x_optimal), steps


a = 1
b = 5


x_optimal, f_optimal, steps = golden_section_search(f, a, b)
print(f"Βέλτιστη τιμή x: {x_optimal}")
print(f"Ελάχιστη τιμή f(x): {f_optimal}")


#for i, (a, b) in enumerate(steps):
  #  print(f"Βήμα {i + 1}: Διάστημα [{a}, {b}]")
