import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def numeric_test(r,d_1,n, tests = 1e5):
    initial_point = np.zeros(n)
    initial_point[0] = -d_1

    t = 0
    success = 0
    trials = 0
    while t < tests:
        trials += 1
        v = np.random.normal(0, 1, size = n)
        v *= r/np.linalg.norm(v)

        if (d_2 := np.linalg.norm(initial_point+v)) < d_1 and d_2 > r/2:
            s = True
            while s:
                u = np.random.normal(0,1,size = n)
                u *= r/np.linalg.norm(u)
                if (d_3 := np.linalg.norm(initial_point+v+u)) < d_2:
                    t += 1
                    s = False
                    if np.linalg.norm(initial_point+u) <= d_3:
                        success += 1
        elif trials > 10*tests**2:
            raise Warning("Something is wrong yo")
    return success / tests


def plot_numerical(n = 10):
    l = 50
    d_1 = 1
    rs = np.linspace(0.01,0.7, l)

    test_data = np.zeros(l)
    # analytical_data = np.zeros(l)
    for i,r in enumerate(rs):
        test_data[i] = numeric_test(r,d_1,n, tests = 10000)

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



if __name__ == "__main__":
    plot_numerical(n = 10)
    # print(numeric_test(0.1,1,2))
    plt.show()