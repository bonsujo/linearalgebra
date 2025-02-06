###Josephine Bonsu 
### January 21, 2025



import numpy as np

#Algorithm 1 - Forward Substitution
def forward_sub(L, b):
    n = len(b)
    x = np.zeros(n)
    for j in range(n):
        x[j] = b[j] / L[j, j]
        for i in range(j + 1, n):
            b[i] -= L[i, j] * x[j]
    return x
#Algorithm 2 - Backward Substitution
def backward_sub(U, b):
    n = len(b)
    x = np.zeros(n)
    for j in range(n - 1, -1, -1):
        x[j] = b[j] / U[j, j]
        for i in range(j):
            b[i] -= U[i, j] * x[j]
    return x
#Algorithm 3 - Gaussian Elimanation
def gauss_elimination(A, b):
    n = len(b)
    for k in range(n - 1):
        for i in range(k + 1, n):
            m = A[i, k] / A[k, k]
            for j in range(k, n):
                A[i, j] -= m * A[k, j]
            b[i] -= m * b[k]
        print(f"Step {k + 1} \nmatrix a = \n{A}\nvector b = {b}\n")
    return A, b

# Defining the system of equations
A = np.array([
    [1, 2, 1, -1],
    [3, 2, 4, 4],
    [4, 4, 3, 4],
    [2, 0, 1, 5]
], dtype=float)
b = np.array([5, 16, 22, 15], dtype=float)

#Gaussian elimination
U, b_transformed = gauss_elimination(A.copy(), b.copy())

# Solving upper triangular system using backward substitution
x = backward_sub(U, b_transformed)

print(f"Solution of x: \n {x}")

