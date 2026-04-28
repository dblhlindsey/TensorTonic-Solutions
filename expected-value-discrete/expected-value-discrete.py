import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    x = np.asarray(x)
    p = np.asarray(p)
    if np.sum(p) != 1:
        raise ValueError("Probabilities must sum to 1")
    else:
        return x@p
