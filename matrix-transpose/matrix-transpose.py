import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    A = np.asarray(A)
    col, row = A.shape
    t = np.zeros((row, col))
    for i in range(row):
        for j in range(col):
            t[i,j] = A[j,i]
    return t