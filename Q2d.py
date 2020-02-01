import numpy as np

def factorial(n):
    num = 1

    if (n != 0):
    	num = n * factorial(n-1)

    return(num)

print(factorial(3))

