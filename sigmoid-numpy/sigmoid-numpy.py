import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    y = 1/(1+np.exp(-1*np.asarray(x, dtype=float)))
    return y
    
x = [0,2,-2]
print(sigmoid(x))