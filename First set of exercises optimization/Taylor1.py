import math

#υπολογίζoyme την προσέγγιση της συνάρτησης f(x) = x * sin(x^2) με τη σειρά Taylor
def taylor_approximation(x, n):
    result = 0
    for i in range(n):
        term = ((-1) ** i) * (x ** (4 * i + 1)) / math.factorial(4 * i + 1)
        result += term
    return result


def real_function(x):
    return x * math.sin(x**2)


x_values = [i / 10 for i in range(-40, 41)]  


n_terms = 5 
for x in x_values:
    taylor_result = taylor_approximation(x, n_terms)
    real_result = real_function(x)
    print(f"x = {x:.1f}: Taylor = {taylor_result:.4f}, Real = {real_result:.4f}")
