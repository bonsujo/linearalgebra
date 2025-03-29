import numpy as np


def newton_A(a, p, r, n0, tol=1e-6, max_iter=100):
    def f(n):
        return a * (1 + r) ** n - p * ((1 + r) ** n - 1) / r
    
    def df(n):
        return a * (1 + r) ** n * np.log(1 + r) - p * ((1 + r) ** n * np.log(1 + r)) / r
    
    n = n0
    print("Iteration | n (years) | X - Y")
    for i in range(max_iter):
        fn = f(n)
        dfn = df(n)
        print(f"{i+1:^9} | {n:^9.4f} | {fn:^9.4f}")
        
        if abs(fn) < tol:
            print(f"Loan will be paid off in {n:.4f} years.")
            print("")
            return n
        
        n = n - fn / dfn

    return n

def newton_B(g, d, x0, y0, tol=1e-6, max_iter=100):
    def F(x, y):
        return np.array([
            g * x * y - x * (1 + y),
            -x * y + (d - y) * (1 + y)
        ])
    
    def J(x, y):
        return np.array([
            [g * y - (1 + y), g * x - x],
            [-y, -x - (1 + 2*y)]
        ])
    
    print("Iteration |  x  |   y   |  f1  |  f2  |")
    x, y = x0, y0
    for i in range(max_iter):
        F_val = F(x, y)
        J_val = J(x, y)
        
        delta = np.linalg.solve(J_val, -F_val)
        x, y = x + delta[0], y + delta[1]
        
        print(f"{i+1:^9} {x:.4f}  {y:.4f} {F_val[0]:.4f} {F_val[1]:.4f}")
        
        if np.linalg.norm(F_val, ord=2) < tol:
            print(f"Solution converged: x = {x:.4f}, y = {y:.4f}")
            return x, y

    return x, y

#challenge 1
a = 100000
p = 10000
r = 0.06
n0 = 10
newton_A(a, p, r, n0)

#challenge 2
g = 5
d = 1
x0, y0 = 7, 0.5
newton_B(g, d, x0, y0)
