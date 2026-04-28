import numpy as np

def bernoulli_pmf_and_moments(x, p):
    """
    Compute Bernoulli PMF and distribution moments.
    """
    pmf = np.array(x, dtype=float)
    pmf = pmf*p-pmf+(pmf-1)*p+1
    mean = p
    var = p*(1-p)
    return pmf, mean, var