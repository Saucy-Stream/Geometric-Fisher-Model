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
    r_values = np.linspace(0, 2/3, 100)  # Values for r between 0 and 2/3
    n_values = np.arange(2, 1000)  # Values for n between 2 and 1000

    # Create a matrix to store the probabilities
    heatmap_data = np.zeros((len(n_values), len(r_values)))

    # Calculate the probability for each combination of r and n
    d_1 = 1  # Define a value for d_1, e.g., 1 (can be adjusted depending on the problem)
    for i, r in enumerate(r_values):
        for j, n in enumerate(n_values):
            heatmap_data[j, i] = analytical_probability(r, d_1, n)

    # Apply logarithmic scale to the heatmap data
    heatmap_data = np.log(heatmap_data + 1e-20)  # Adding a small value to avoid log(0)

    # Create heatmap
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, cmap='viridis', cbar_kws={'label': 'Log Probability'})

    plt.xticks(ticks=np.arange(0, len(r_values), step=10), labels=np.round(r_values[::10], 2))
    plt.yticks(ticks=np.arange(0, len(n_values), step=50), labels=n_values[::50])
    plt.gca().invert_yaxis()
    plt.title("Logarithmic Scale Analytical Probability Heatmap")
    plt.xlabel('r/d Values')
    plt.ylabel('n Values')
    plt.savefig("Figures/duplication_heatmap_log")
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