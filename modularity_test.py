import numpy as np
import matplotlib.pyplot as plt

def numeric_test(n, tests = 1e5):
    mods = np.zeros(tests)
    for i in range(tests):
        v = np.random.normal(0,1,n)
        # print(f"Vector: {v} \nProportion modularity: {proportion_modularity(v)} \nStrongest direction modularity: {strongest_direction_modularity(v)}\n")
        modularity = strongest_direction_modularity(v)
        mods[i] = modularity
    return mods

def plot_numerical(ns = np.arange(2,10), tests = 1000):
    l = len(ns)
    test_data = np.zeros((l,tests))
    analytical_data = np.zeros(l)
    for i,n in enumerate(ns):
        test_data[i] = numeric_test(n, tests)
        analytical_data[i] = (np.sqrt(2*np.log(n))-1)/(np.sqrt(n)-1)

    fig = plt.figure(figsize = (10,8))
    ax = plt.subplot()
    ax.boxplot(test_data.T, positions = ns)
    ax.plot(ns,analytical_data, color = 'r')
    ax.plot(ns,np.mean(test_data,1))
    ax.grid()
    ax.legend()
    return fig

def strongest_direction_modularity(vector: np.ndarray[float]) -> float:
        n = len(vector)  # Dimension of the vectors
        scale = 1 / np.sqrt(n)
        vec = np.copy(vector)
        vec /= np.linalg.norm(vec)
        strongest_dir = max(abs(vec))
        modularity = (strongest_dir - scale) / (1 - scale)

        return modularity

def proportion_modularity(vector: np.ndarray[float]) -> float:
        vec = abs(np.copy(vector))

        next_strongest,strongest = np.partition(vec,-2)[-2:]
        modularity = 1 - next_strongest/strongest

        return modularity

if __name__ == "__main__":
    # numeric_test(n = 30, tests = 5)
    ns = np.arange(10,11)
    plot_numerical(ns)
    # print(analytical_probability(0.5,1,2))
    plt.show()