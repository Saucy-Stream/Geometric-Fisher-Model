import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def numeric_test(r,d_1,n, tests = 1e5):
    initial_point = np.zeros(n)
    initial_point[0] = d_1

    t = 0
    success = 0
    while t < tests:
        v = np.random.normal(0,1,n)
        v *= r/np.linalg.norm(v)
        if (d_2 := np.linalg.norm(initial_point+v)) <= d_1:
            t += 1
            if np.linalg.norm(initial_point+2*v) <= d_2:
                success += 1
    return success / tests

def analytical_probability(r, d_1, n):
    if r >= 2/3*d_1:
        return 0
    N = 1000

    x_1 = np.linspace(0, np.arccos(r / (2 * d_1)), N)
    y_1 = np.sin(x_1)**(n - 2)
    beneficial = np.trapezoid(y_1, x_1)

    x_2 = np.linspace(0, np.arccos(3 * r / (2 * d_1)), N)
    y_2 = np.sin(x_2)**(n - 2)
    duplication = np.trapezoid(y_2, x_2)

    return duplication / beneficial


def plot_probability_heatmap():
    r_values = np.linspace(0, 2/3, 100)  # Värden för r mellan 0 och 2/3
    n_values = np.arange(2, 1000)  # Värden för n mellan 2 och 100

    # Skapa en matris för att lagra sannolikheterna
    heatmap_data = np.zeros((len(n_values), len(r_values)))

    # Beräkna sannolikheten för varje kombination av r och n
    d_1 = 1  # Definiera ett värde för d_1, t.ex. 1 (kan justeras beroende på problem)
    for i, r in enumerate(r_values):
        for j, n in enumerate(n_values):
            heatmap_data[j, i] = analytical_probability(r, d_1, n)

    # Skapa heatmap
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, cmap='viridis', cbar_kws={'label': 'Probability'})


    plt.xticks(ticks=np.arange(0, len(r_values), step=10), labels=np.round(r_values[::10], 2))
    plt.yticks(ticks=np.arange(0, len(n_values), step=50), labels=n_values[::50])
    plt.gca().invert_yaxis()
    plt.title("Analytical Probability Heatmap")
    plt.xlabel('r/d Values')
    plt.ylabel('n Values')
    plt.savefig("Figures/duplication_heatmap")
    return fig

def plot_numerical(n = 10):
    l = 100
    d_1 = 1
    rs = np.linspace(0,2/3, l)

    test_data = np.zeros(l)
    analytical_data = np.zeros(l)
    for i,r in enumerate(rs):
        test_data[i] = numeric_test(r,d_1,n, tests = 1000)
        analytical_data[i] = analytical_probability(r,d_1,n)

    fig = plt.figure(figsize = (10,8))
    ax = plt.subplot()
    ax.plot(rs,analytical_data, c = "k", label = "Analytic solution")
    ax.plot(rs,test_data, c = "r", label = "Numerical data")
    ax.grid()
    ax.legend()
    return fig


if __name__ == "__main__":
    # plot_numerical(n = 100)
    plot_probability_heatmap()

    # print(analytical_probability(0.5,1,2))
    plt.show()