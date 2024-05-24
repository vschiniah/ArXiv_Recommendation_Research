import numpy as np
from math import pow

# default ranking 1 > 2 > ... > n
def get_true_ranking(n):
    return np.arange(1,n+1)

def sample_mallows(phi, sigma=None, n=2):
    # if no ranking sigma is provided, use Mallows(phi, 1 > 2 > ... > n)
    if sigma is None:
        return sample_mallows_rim(phi, get_true_ranking(n))
        
    # otherwise, sample from Mallows(phi, sigma)
    return sample_mallows_rim(phi, sigma)

# sample from Mallows model (see Lu, Tyler and Craig Boutilier. "Effective Sampling and Learning for Mallows Models with Pairwise-Preference Data". Journal of Machine Learning Research, 2014.)
def sample_mallows_rim(phi, sigma):
    rng = np.random.default_rng()

    n = len(sigma)
    r = []
    prob_factors = []
    for i in range(1,n+1):
        prob_factors.insert(0, pow(phi,i-1))

        if i == 1:
            j = 1
        else:
            j = rng.choice(np.arange(1, i+1), p= prob_factors / np.sum(prob_factors))
        r.insert(j-1, sigma[i-1])

    return r

def get_mallows_utilities(m,n,phi):
    U = np.zeros((m,n))
    for user in range(m):
        # sample a ranking according to Mallows(phi, 1 > 2 > ... > n)
        r = sample_mallows(phi, n=n)

        # compute utility based on ranking (linear decay based on rank)
        U[user,:] = 1 - np.argsort(r) / n

    return U