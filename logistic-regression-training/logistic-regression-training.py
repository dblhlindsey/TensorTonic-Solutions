import numpy as np

def sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return 1/(1+np.exp(-1*np.asarray(z,dtype=float)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    X = np.array(X,dtype=float)
    y = np.array(y,dtype=float)

    N, d = X.shape
    
    b = 0.0
    w = np.zeros(d)
    
    for i in range(steps):
        p = sigmoid(X@w + b)
        w = w - lr*(X.T@(p-y)/N)
        b = b - lr*np.mean(p-y)
        
    return w, b

X = [[0],[1],[2],[3]]
y = [0,1,1,1]

train_logistic_regression(X, y)