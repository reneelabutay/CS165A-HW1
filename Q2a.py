import numpy as np

# A: 5x4 matrix 1,2,...,20
A = np.arange(start=1, stop=21).reshape(5, 4)
print(A)

# B: 4x3 matrix 1,2,...12
B = np.arange(start=1, stop=13).reshape(4, 3)
print(B)

C = np.dot(A, B)
print(C)