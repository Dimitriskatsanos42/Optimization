import sympy as sp

x = sp.symbols('x')

f_x = sp.sin(x)

taylor_series = sp.series(f_x, x, 0, 5)

a = -4
b = 4

for x_val in range(a, b+1):
    taylor_approximation = taylor_series.subs(x, x_val)
    print(f"x = {x_val}: Taylor = {taylor_approximation}")
