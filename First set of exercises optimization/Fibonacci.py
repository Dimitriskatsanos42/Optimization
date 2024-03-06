import math

def f(x):
    return x * (x - 3)**2 + 2


def fibonacci_search(f, a, b, tol=1e-6):
    n = 1
    while (b - a) > tol:
        
        x1 = a + (F(n - 2) / F(n)) * (b - a) 
        x2 = a + (F(n - 1) / F(n)) * (b - a)
        
        
        f1 = f(x1)
        f2 = f(x2)
        

        if f1 < f2:
            b = x2
        else:
            a = x1
        
        n += 1
    
    x_optimal = (a + b) / 2
    return x_optimal, f(x_optimal)


def F(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


a = 2
b = 4


x_optimal, f_optimal = fibonacci_search(f, a, b)
print(f"Βέλτιστη τιμή x: {x_optimal}")
print(f"Ελάχιστη τιμή f(x): {f_optimal}")
