import numpy as np
import scipy as sc
from scipy.stats import ncx2
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from Test_files.duplication_test import analytical_probability as duplication_probability


def analytical_reject_prob(n,d,sigma):
    lam = (d/sigma)**2
    chi_sq = ncx2(n,lam)
    prob = chi_sq.cdf(lam)

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
    chi_sq = ncx2(n,lam)

    prob = chi_sq.cdf(lam)
    rejection = d*(1-prob)

    def beneficial_prob(x):
        prob = np.sqrt(x)*chi_sq.pdf(x)
        if np.isnan(prob):
            return 0
        return prob

    beneficial = sigma*integrate.quad(beneficial_prob,0,lam)[0]
    return rejection+beneficial

def test_expectation(n,d,sigmas,tests = 1000):
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

def analytical_simulation(n,initial_distance,sigma,timestop, duplication_rate = 0.02, initial_nb_genes : float = 1, plot = False):
    """
    Finds the expected distance to the optimum for a given number of generations

    Parameters:
    n : int
        The number of genes
    initial_distance : float
        The initial distance to the optimum
    sigma : float
        The standard deviation of the mutations
    timestop : int
        The number of generations to simulate
    duplication_rate : float
        The rate of duplication of the genes
    initial_nb_genes : float
        The initial number of genes
    plot : bool
        Whether to plot the results or not
    
    Returns:
    distances : np.array
        The expected distance to the optimum at each generation
    nb_genes : np.array
        The number of genes at each generation
    """

    with_duplication = (duplication_rate > 0)
    distances = np.zeros(timestop)
    distances[0] = initial_distance
    nb_genes : np.ndarray[float] = np.full(timestop,initial_nb_genes, dtype=float)
    if with_duplication:
        for t in range(1,timestop):
            distances[t] = max(analytical_expectation(n,distances[t-1],sigma*np.sqrt(nb_genes[t-1])),0)
            average_gene_size = (initial_distance-distances[t-1]) / nb_genes[t-1]
            new_genes = nb_genes[t-1]*duplication_rate*duplication_probability(average_gene_size,distances[t-1]-average_gene_size,n)
            distances[t] = max(distances[t]-average_gene_size*new_genes,0)
            nb_genes[t] = nb_genes[t-1]+new_genes
            print(f"Analytical simulation for n = {n}, d = {initial_distance} at time {t}/{timestop}", end = "\r")
    else:
        for t in range(1,timestop):
            distances[t] = analytical_expectation(n,distances[t-1],sigma*np.sqrt(initial_nb_genes))
            print(f"Analytical simulation for n={n}, d = {initial_distance} at time {t}/{timestop}", end = "\r")
    print(f"Analytical simulation for n = {n}, d = {initial_distance} until t = {timestop} done", end = "\033[K\n")

    if plot:
        fig = plt.figure(figsize = (10,4))
        ax = fig.subplots()
        ax.grid()
        ax.plot(range(timestop),distances, label = "Expected distance to the optimum")
        ax.set_xlabel("Time")
        ax.set_ylabel("Distance")
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_title(f"Expected distance to the optimum with starting values n = {n}, d = {d}, sigma = {sigma}")
        fig.savefig("Figures/analytical_distance_to_the_optimum")
    return distances, nb_genes

if __name__ == "__main__":
    n = 10
    d = 10
    sigmas = np.linspace(0.01,d/2,1000)
    s = 0.1

    # nx = ncx2(n,d/2)
    # fig = plt.figure(figsize = (10,4))
    # ax = fig()
    # test_rejections(n,d)
    # test_expectation(n,d,sigmas)
    analytical_simulation(n,d,s, int(1e4))
    plt.show()
