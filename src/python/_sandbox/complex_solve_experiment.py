import numpy as np
from numpy.linalg.linalg import solve
import time

N = 100000

A = np.array([
    [1.0, 0.0, 0.0, 2.+5j],
    [0.0, 1.0+2.j, 0.0, 0.0],
    [0.0, 3.j, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])

b = np.array([2.0, 3.0, 4.0, 5.0])

print(A)
print(b)
s = time.time()
for i in range(N):
    r = solve(A, b)
print(time.time() - s)
print(r)
print(r.real, r.imag)
print("-" * 20)

big_A = np.zeros(shape=(2*len(A), 2*len(A)))
A0 = np.s_[0:len(A)]
A1 = np.s_[len(A):2*len(A)]
big_A[A0, A0] = A.real
big_A[A1, A0] = A.imag
big_A[A0, A1] = -A.imag
big_A[A1, A1] = A.real

big_b = np.zeros(shape=(2*len(b)))
big_b[0:len(b)] = b.real
big_b[len(b):2*len(b)] = b.imag

print(big_A)
print(big_b)
s = time.time()
for i in range(N):
    r = solve(big_A, big_b)
print(time.time() - s)
print(r)
print(r[0:len(b)], r[len(b):2*len(b)])
print("-" * 20)
