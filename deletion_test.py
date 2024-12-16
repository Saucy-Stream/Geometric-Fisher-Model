import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def numeric_test(r,d_1,n, tests = 1e5):
    initial_point = np.zeros(n)
    initial_point[0] = -d_1

    t = 0
    success = 0
    proportion = 1

    min_size = proportion*r/2
    max_trials = 0
    while t < tests:
        v = abs(np.random.normal(0, 1, size = n))
        v *= r/np.linalg.norm(v)

        if (d_2 := np.linalg.norm(initial_point+v)) < d_1 and d_2 > min_size:
            s = True
            trials = 0
            while s:
                trials += 1
                u = np.random.normal(0,1,size = n)
                u *= r/np.linalg.norm(u) * proportion
                if (d_3 := np.linalg.norm(initial_point+v+u)) < d_2:
                    t += 1
                    s = False
                    if np.linalg.norm(initial_point+u) <= d_3:
                        success += 1
                    if trials > max_trials:
                        max_trials = trials
                if trials > 1000000:
                    s = False
                    print("Trials abandoned")
    # print(f"n: {n}, r: {r}, max trials: {max_trials}, succ_prob = {success/tests}")
    return success / tests


def plot_numerical(n = 10):
    l = 50
    d_1 = 1
    rs = np.linspace(0.01,0.9, l)

    test_data = np.zeros(l)
    analytical_data = np.zeros(l)
    for i,r in enumerate(rs):
        test_data[i] = numeric_test(r,d_1,n, tests = 10000)
        analytical_data[i] = analytical_sol(n,r)
        print(f"r = {r} done")

    fig = plt.figure(figsize = (10,8))
    ax = plt.subplot()
    ax.plot(rs,analytical_data, c = "k", label = "Analytic solution")
    ax.plot(rs,test_data, c = "r", label = "Numerical data")
    ax.set_xlabel("Size of genes (r/d)")
    ax.set_ylabel("Proportion of beneficial deletions")
    ax.set_title(f"Numerical probability of deletion in dimension n = {n}")
    ax.grid()
    ax.legend()
    return fig

def ben(r,d):
    return np.arccos(r/(2*d))

def del_prob(th,r,n):
    l = 100
    ths = np.linspace(0,th,l)
    y1 = np.sin(ths)**(n-2)

    nom = np.trapezoid(y1,ths, axis = 0)

    d = np.sqrt(r*r+1-2*r*np.cos(th))
    phmax = ben(r,d)
    phis = np.linspace(0,phmax)

    int2 = np.sin(phis)**(n-2)
    den = np.trapezoid(int2,phis, axis = 0)
    prob = nom/den
    return prob

def analytical_sol(n = 5, r = 0.1):
    thmax = np.arccos(r/2)
    thdup = np.arccos(3*r/2)

    l = 1000
    ths = np.linspace(0,thmax,l)
    split = np.searchsorted(ths, thdup)

    #inside beneficial duplication range:

    th1 = ths[:split]
    ys = del_prob(th1,r,n)
    prob = np.trapezoid(ys,th1)




    #outside beneficial duplication range:
    # th2 = ths[split:]
    # d2 = np.sqrt(r*r+1+2*r*np.cos(th2))
    # possible_ph_2 = 2*ben(r,d2)
    # y_denom = np.sin(ths)**(n-2)
    return prob

def plot_probability_heatmap():
    r_values = np.linspace(0.01, np.sqrt(2), 30)
    n_values = np.arange(2, 22,2)

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
    # analytical_sol(2)

    # print(del_prob(np.pi/3,0.01,2))

    plt.show()