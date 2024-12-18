import numpy as np
import scipy as sc
from scipy.stats import ncx2
import matplotlib.pyplot as plt


def analytical_reject_prob(n,d,sigma):
    lam = (d/sigma)**2
    n_chi_squared_obj = ncx2(n,lam)
    prob = n_chi_squared_obj.cdf(lam)

    return 1-prob

def numeric_reject_prob(n,d,sigma,tests = 1000):
    P = np.zeros(n)
    P[0] = d
    rejections = 0
    for i in range(tests):
        v = np.random.normal(0,sigma,n)
        if np.linalg.norm(P+v) > d:
            rejections += 1
    
    return rejections / tests

def test_rejections(n,d,tests = 1000):
    l = 200
    sigmas = np.linspace(0.1,10,l)
    analytical_prob = np.zeros(len(sigmas))
    numeric_prob = np.zeros(len(sigmas))
    for i,s in enumerate(sigmas):
        analytical_prob[i] = analytical_reject_prob(n,d,s)
        numeric_prob[i] = numeric_reject_prob(n,d,s,tests)

    fig = plt.figure(figsize = (10,4))
    ax = fig.subplots()
    ax.set_xlabel("Variance")
    ax.set_ylabel("Probability")
    ax.plot(sigmas,analytical_prob, label = "Analytical rejection probability")
    ax.plot(sigmas,numeric_prob, label = "Numeric rejection probability")
    
    ax.grid()
    ax.legend()
    plt.show()

def numeric_expectation(n,d,sigma,tests):
    P = np.zeros(n)
    P[0] = d
    values = np.zeros(tests)
    for i in range(tests):
        v = np.random.normal(0,sigma,n)
        if np.linalg.norm(P+v) > d:
            values[i] = d
        else:
            values[i] = np.linalg.norm(P+v)
    
    return np.mean(values)

def analytical_expectation(n,d,sigma):
    lam = (d/sigma)**2
    n_chi_squared_obj = ncx2(n,lam)
    prob = n_chi_squared_obj.cdf(lam)

    rejection = d*(1-prob)
    
    xs = np.linspace(0,(d/sigma)**2)
    ys = sigma*np.sqrt(xs)*n_chi_squared_obj.pdf(xs)

    beneficial = np.trapezoid(ys,xs)
    return rejection+beneficial

def test_expectation(n,d,tests = 1000):
    l = 200
    sigmas = np.linspace(0.2,10,l)
    analytical_exp = np.zeros(len(sigmas))
    numeric_exp = np.zeros(len(sigmas))
    for i,s in enumerate(sigmas):
        analytical_exp[i] = analytical_expectation(n,d,s)
        numeric_exp[i] = numeric_expectation(n,d,s,tests)

    fig = plt.figure(figsize = (10,4))
    ax = fig.subplots()
    ax.set_xlabel("Variance")
    ax.set_ylabel("Expected distance to the origin of next generation")
    ax.plot(sigmas,analytical_exp, label = "Analytical expectation")
    ax.plot(sigmas,numeric_exp, label = "Numeric expectation")
    
    ax.set_ylim(0,d*1.1)
    ax.grid()
    ax.legend()
    plt.show()

if __name__ == "__main__":
    n = 5
    d = 20
    # nx = ncx2(n,d/2)
    # fig = plt.figure(figsize = (10,4))
    # ax = fig()
    # test_rejections(n,d)
    test_expectation(n,d)
