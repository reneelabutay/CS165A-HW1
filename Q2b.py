# Write a python
# function that takes a nonnegative integer n and outputs a sparse matrix A of size n − 1 × n,
# such that for any x ∈ R^n
# , Ax = [x1−x2, ..., xn−1−xn]^T
# Write a script that calls this function
# and print the resulting A for n = 5. (Make sure you use the sparse matrix library in python
# is scipy.sparse.)


import numpy as np
from scipy.sparse import csr_matrix

def generate(n):
    data = np.zeros(shape=(n-1,n))
    i = 0
    while i < n-1:
        data[i][i] = 1.0
        data[i][i+1] = -1.0
        i += 1
    a = csr_matrix(data)
    return a

A = generate(5)
print(A)

"""
Output example:
  (0, 0)	1.0
  (0, 1)	-1.0
  (1, 1)	1.0
  (1, 2)	-1.0
  (2, 2)	1.0
  (2, 3)	-1.0
  (3, 3)	1.0
  (3, 4)	-1.0
"""