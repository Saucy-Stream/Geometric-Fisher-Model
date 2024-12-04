import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def numeric_test(r,d_1,n, tests = 1e5):
    initial_point = np.zeros(n)
    initial_point[0] = -d_1

    t = 0
    success = 0
    trials = 0
    proportion = 2
    while t < tests:
        trials += 1
        v = np.random.normal(0, 1, size = n)
        v *= r/np.linalg.norm(v)

        if (d_2 := np.linalg.norm(initial_point+v)) < d_1 and d_2 > proportion*r/2:
            s = True
            while s:
                u = np.random.normal(0,1,size = n)
                u *= r/np.linalg.norm(u) * proportion
                if (d_3 := np.linalg.norm(initial_point+v+u)) < d_2:
                    t += 1
                    s = False
                    if np.linalg.norm(initial_point+u) <= d_3:
                        success += 1

    return success / tests


def plot_numerical(n = 10):
    l = 50
    d_1 = 1
    rs = np.linspace(0.01,0.9, l)

    test_data = np.zeros(l)
    # analytical_data = np.zeros(l)
    for i,r in enumerate(rs):
        test_data[i] = numeric_test(r,d_1,n, tests = 10000)
        print(f"r = {r} done")

    fig = plt.figure(figsize = (10,8))
    ax = plt.subplot()
    # ax.plot(rs,analytical_data, c = "k", label = "Analytic solution")
    ax.plot(rs,test_data, c = "r", label = "Numerical data")
    ax.set_xlabel("Size of genes (r/d)")
    ax.set_ylabel("Proportion of beneficial deletions")
    ax.set_title(f"Numerical probability of deletion in dimension n = {n}")
    ax.grid()
    ax.legend()
    return fig

def plot_probability_heatmap():
    r_values = np.linspace(0.01, 0.7, 100)
    n_values = np.arange(2, 32,2)

    heatmap_data = np.zeros((len(n_values), len(r_values)))

    d_1 = 1
    for i, n in enumerate(n_values):
        for j, r in enumerate(r_values):
            heatmap_data[i, j] = numeric_test(r, d_1, n, tests = 500)
        print(f"Dimension {n} done")

    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, cmap='viridis', cbar_kws={'label': 'Probability'})


    plt.xticks(ticks=np.arange(0, len(r_values), step=10), labels=np.round(r_values[::10], 2))
    plt.yticks(ticks=np.arange(0, len(n_values), step=5), labels=n_values[::5])
    plt.gca().invert_yaxis()
    plt.title("Analytical Probability Heatmap")
    plt.xlabel('r/d Values')
    plt.ylabel('n Values')
    plt.savefig("Figures/deletion_heatmap")
    return fig


if __name__ == "__main__":
    # plot_numerical(n = 2)
    # print(numeric_test(0.1,1,2))
    plot_probability_heatmap()
    plt.show()