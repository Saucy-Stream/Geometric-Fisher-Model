from GFGM_model import FisherGeometricModel
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pickle
import json
import copy
from Test_files.mutation_test import analytical_simulation
from Test_files.duplication_test import analytical_probability
import seaborn as sns
import os

def plot_vizualised_path(fgm : FisherGeometricModel) -> None:
    """
    Plots the final phenotype and underlying genotype of a 2- or 3-dimensional FisherGeometricModel
    
    Parameters
    -----
    fgm: FisherGeometricModel
        The FisherGeometricModel to be visualized
    
    Returns
    -----
    None
        Plots path
    """

    
    if fgm.dimension == 2:
        fig = plt.figure(figsize = (10,4))
        ax = fig.add_subplot()
        for radius in [2,5,10,15,20,30]:
            circ = plt.Circle((0,0),radius, fill = False, color = 'k')
            ax.add_patch(circ)
        kwargs = {'scale_units' : 'xy', 'angles' : 'xy', 'color': 'k', 'scale' : 1}

    elif fgm.dimension == 3:
        fig = plt.figure(figsize = (10,4))
        ax = fig.add_subplot(projection = '3d')
        kwargs = {'color': 'k'}
    else:
        print("Unable to plot a visualization graph when dimensions > 3")
        return
    
    unique_positions = np.array([pos for i,pos in enumerate(fgm.positions) if sum(fgm.methods[i])>0])

    prev_pos = unique_positions[0]
    for pos in unique_positions[1:]:
        line = np.array([prev_pos,pos]).T
        ax.plot(*line, c = 'r')
        prev_pos = pos

    ax.scatter(*unique_positions.T,c = "r", label = 'Evolutionary path')

    number_of_genes = np.shape(fgm.genes)[0]
    genome = np.resize(fgm.genes.copy(),(number_of_genes,fgm.dimension))

    position = np.array(fgm.init_pos)
    ax.scatter(*position, c = 'r')
    for gene in genome:
        ax.quiver(*position,*gene,**kwargs)
        position += gene
        ax.scatter(*position, c = 'g')
    
    origin = fgm.optimum

    ax.scatter(*origin, c = 'k')
    ax.grid()
    ax.legend(fontsize=12)
    return fig

def draw_phenotype_size(fgm : FisherGeometricModel):
    fig = plt.figure(figsize = (10,4))
    ax = plt.subplot()
    
    genes = abs(fgm.genes)
    a = np.max(genes,axis = 0)
    # genes /= a
    ax.hist(genes, bins = 'auto', stacked = True)

    return fig


def draw_distance_plot(fgms: list[FisherGeometricModel], with_analytical = False):
    fig = plt.figure(figsize = (10,4))
    ax = fig.subplots()
    
    for i,fgm in enumerate(fgms):
        y2 = np.linalg.norm(fgm.positions, axis=1)
        x = np.arange(fgm.current_time+1)

        ax.plot(x,y2, label = f'Run {i+1}')
        if with_analytical:
            args = fgm.get_args()
            if "duplication" in args["mutation_methods"]:
                with_duplication = True
            else:
                with_duplication = False
            duplication_rate = args["duplication_rate"] if with_duplication else 0
            analytical_sol = analytical_simulation(args["n"], args["initial_distance"], args["sigma_add"],fgm.current_time+1, with_duplication=with_duplication, duplication_rate=duplication_rate)
            ax.plot(x,analytical_sol, label = f"Analytical solution {(not with_duplication)*'(no duplication) '}for run {i+1}")

    ax.set_ylabel("Distance", fontsize=14)
    ax.set_xlabel("Time", fontsize=14)
    ax.set_yscale('log')
    ax.set_xscale("log")
    ax.grid()
    ax.tick_params(axis='both', which='major', labelsize=12)

    ax.legend(fontsize=12)
    fig.tight_layout()
    fig.savefig("Figures/distance_to_optimum")
    return fig

def compare_runs_with_reset(fgm_args: dict, reset_points: list[int], total_generations: int, nb_runs: int = 100):
    """
    Test the effect of genome size with multiple resets and visualize them in the same plot.
    
    Parameters:
    - fgm_args: dict, arguments for initializing the FisherGeometricModel
    - reset_points: list[int], list of generations at which to reset the genome
    - total_generations: int, number of generations to simulate after each reset
    
    Returns:
    - fig: matplotlib figure, the comparison plot
    """
    if os.path.exists("test_data/multiple_resets_comparison.pkl"):
        used_args, all_runs = load_simulation_data("test_data/multiple_resets_comparison.pkl")
    else:
        used_args = fgm_args.copy()
        total_generations = int(total_generations)
        used_args['display_fixation'] = False
        used_args['mutation_methods'] = ["addition", "duplication"]
        used_args['duplication_rate'] = 0.02
        used_args['initial_distance'] = 100
        used_args["n"] = 100

        resets = len(reset_points)
        
        all_runs = np.zeros((nb_runs,len(reset_points)+1,total_generations+2))
        for run in range(nb_runs):
            current_generations = 0
            fgm = FisherGeometricModel(**used_args)

            for i,reset in enumerate(reset_points):
                reset = int(reset)
                fgm.evolve_successive(reset - current_generations)
                current_generations = reset

                fgm_2 = copy.deepcopy(fgm)
                fgm_2.reinitialize()

                remaining_generations = total_generations-reset
                fgm_2.evolve_successive(remaining_generations)

                all_runs[run,i] = np.linalg.norm(fgm_2.positions, axis=1)
            fgm.evolve_successive(remaining_generations)
            all_runs[run,-1] = np.linalg.norm(fgm.positions, axis=1)
            print(f"Reset run {run+1}/{nb_runs} completed", end="\r")
        print("All reset runs completed!")
        used_args["total_generations"] = total_generations
        used_args["reset_points"] = reset_points
        used_args["nb_runs"] = nb_runs
        save_simulation_data("test_data/multiple_resets_comparison.pkl", (used_args, all_runs))
    total_generations = used_args["total_generations"]
    means = np.mean(all_runs, axis=0)
    percentiles = np.percentile(all_runs,[20,80], axis=0)
    
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlabel("Time", fontsize=14)
    ax.set_ylabel("Distance to the Optimum", fontsize=14)
    ax.set_yscale('log')
    # ax.set_xscale("log")
    ax.grid()
    ax.tick_params(axis='both', which='major', labelsize=12)
    x = np.arange(total_generations+2)
    for i in range(len(reset_points)):
        m = means[i]
        lower = percentiles[0][i]
        upper = percentiles[1][i]
        ax.plot(x, m, label=f"Reset at generation {reset_points[i]}")
        ax.fill_between(x, lower, upper, alpha=0.3)
    ax.plot(x, means[-1], label="No Reset")
    ax.fill_between(x, percentiles[0][-1], percentiles[1][-1], alpha=0.3)

    ax.legend(fontsize=12)
    print("Used arguments for multiple resets comparison plot:", used_args)
    fig.savefig("Figures/multiple_resets_comparison")
    return fig

def show_simulation_results(fgm : FisherGeometricModel):
    # print(f"Number of genes in time: {fgm.get_nb_genes()}")
    # print(fgm.methods)
    print(f"Number of {fgm.get_args()['mutation_methods']} = {np.sum(fgm.methods,axis = 0)}")
    print(f"Number of genes: {np.unique(fgm.get_nb_genes(), return_counts=True)}")
    # gene_sizes = np.linalg.norm(fgm.genes, axis = 1)
    # print(f"Size of genes: {np.sort(gene_sizes)}")
    print(f"Inital beneficial directions: {fgm.initial_beneficial_directions}")
    # print(f"Genes : {fgm.genes}")
    # print(f"Initial position: {fgm.init_pos}")
    plot_vizualised_path(fgm)
    # # draw_gene_trait_graph(fgm)
    draw_distance_plot([fgm])
    return

def test_analytical_vs_numerical(fgm_args: dict, num_runs: int = 100, time_steps: int = 1000):
    """
    Compare the analytical solution with numerical simulations of the FisherGeometricModel.
    
    Parameters:
    - fgm_args: dict, arguments for initializing the FisherGeometricModel
    - num_runs: int, number of numerical simulations to run
    - time_steps: int, number of time steps for the simulation
    
    Returns:
    - fig: matplotlib figure, the comparison plot
    """
    savefile = "test_data/analytical_solution.pkl"
    if os.path.exists(savefile):
        used_args, with_duplication, analytical_distances, analytical_nb_genes, numerical_distances, numerical_nb_genes = load_simulation_data(savefile)
    else:
        used_args = fgm_args.copy()
        n = used_args['n']

        used_args['display_fixation'] = False
        initial_distance = used_args['initial_distance']
        sigma_add = used_args['sigma_add']
        with_duplication = "duplication" in used_args['mutation_methods']
        duplication_rate = used_args.get('duplication_rate') * with_duplication
        

        # Analytical solution
        analytical_distances, analytical_nb_genes = analytical_simulation(n, initial_distance, sigma_add, time_steps+2, duplication_rate)

        # Numerical simulations
        numerical_distances = np.zeros((num_runs, time_steps+2))
        numerical_nb_genes = np.zeros((num_runs, time_steps+2))
        for run in range(num_runs):
            fgm = FisherGeometricModel(**used_args)
            fgm.evolve_successive(time_steps)
            numerical_distances[run] = np.linalg.norm(fgm.positions, axis=1)
            numerical_nb_genes[run] = fgm.get_nb_genes()
            print(f"Run {run + 1}/{num_runs} completed", end="\r")
        
        used_args["with_duplication"] = with_duplication
        used_args["num_runs"] = num_runs
        used_args["time_steps"] = time_steps
        save_simulation_data(savefile, (used_args, with_duplication, analytical_distances, analytical_nb_genes, numerical_distances, numerical_nb_genes))

    time_steps = used_args["time_steps"]
    if with_duplication:
        # Plotting the results
        fig = plt.figure(figsize=(10, 8))
        ax1, ax2 = fig.subplots(1,2)

        # Plot analytical solution
        x = range(time_steps+2)
        
        percentiles = [10,90]
        mean = np.mean(numerical_distances, axis=0)
        lower = np.percentile(numerical_distances, percentiles[0], axis=0)
        upper = np.percentile(numerical_distances, percentiles[1], axis=0)
        # Plot numerical simulations
        ax1.plot(x, mean, label="Average of Numerical simulations", color="r")
        ax1.fill_between(x, lower, upper, alpha=0.3, color="r", label = f"{percentiles[0]}-{percentiles[1]}% confidence interval")
        ax1.plot(x, analytical_distances, label="Analytical solution", color="k")

        ax1.set_xlabel("Time", fontsize=14)
        ax1.set_ylabel("Distance to the optimum", fontsize=14)
        ax1.set_yscale("log")
        # ax1.set_xscale("log")
        ax1.legend(fontsize=12)
        ax1.grid()
        ax1.tick_params(axis='both', which='major', labelsize=12)


        mean = np.mean(numerical_nb_genes, axis=0)
        lower = np.percentile(numerical_nb_genes, percentiles[0], axis=0)
        upper = np.percentile(numerical_nb_genes, percentiles[1], axis=0)

        ax2.plot(x, mean, label="Average of Numerical simulations", color="r")
        ax2.fill_between(x, lower, upper, alpha=0.3, color="r", label = f"{percentiles[0]}-{percentiles[1]}% confidence interval")
        ax2.plot(x, analytical_nb_genes, label="Analytical solution", color="k")

        ax2.set_xlabel("Time", fontsize=14)
        ax2.set_ylabel("Number of genes", fontsize=14)
        ax2.set_xscale("log")
        ax2.legend(fontsize=12)
        ax2.grid()
        ax2.tick_params(axis='both', which='major', labelsize=12)


        ax1.text(0.93, 0.98, 'a)', transform=ax1.transAxes, fontsize=16, va='top', bbox=dict(facecolor='0.7', edgecolor='none', pad=3.0))
        ax2.text(0.93, 0.98, 'b)', transform=ax2.transAxes, fontsize=16, va='top', bbox=dict(facecolor='0.7', edgecolor='none', pad=3.0))

        fig.savefig("Figures/analytical_vs_numerical_comparison_with_duplication")
    else:

        # Plotting the results
        fig = plt.figure(figsize=(6, 6))
        axs = fig.subplots()

        # Plot analytical solution
        x = range(time_steps+2)

        # Plot numerical simulations
        percentiles = [10,90]
        mean = np.mean(numerical_distances, axis=0)
        lower = np.percentile(numerical_distances, percentiles[0], axis=0)
        upper = np.percentile(numerical_distances, percentiles[1], axis=0)

        axs.plot(x, mean, label="Average of Numerical simulations", color="r")
        axs.fill_between(x, lower, upper, alpha=0.3, color="r", label = f"{percentiles[0]}-{percentiles[1]}% confidence interval")
        axs.plot(x, analytical_distances, label="Analytical solution", color="k")

        axs.set_xlabel("Time", fontsize=14)
        axs.set_ylabel("Distance to the optimum", fontsize=14)
        axs.set_yscale("log")
        # axs.set_xscale("log")
        axs.legend(fontsize=12)
        axs.grid()
        axs.tick_params(axis='both', which='major', labelsize=12)


        fig.savefig("Figures/analytical_vs_numerical_comparison_no_duplication")
    
    return fig


def compare_analytical_numerical_heatmap(fgm_args: dict, n_values: list[int], distance_values: list[float], num_runs: int = 100, time_steps: int = 1000):
    """
    Compare the analytical solution with numerical simulations of the FisherGeometricModel for different values of n and initial distance.
    
    Parameters:
    - fgm_args: dict, arguments for initializing the FisherGeometricModel
    - n_values: list[int], list of different values for the dimension n
    - distance_values: list[float], list of different values for the initial distance to the optimum
    - num_runs: int, number of numerical simulations to run for each combination of n and initial distance
    - time_steps: int, number of time steps for the simulation
    
    Returns:
    - fig: matplotlib figure, the heatmap plot
    """
    savefile = "test_data/analytical_vs_numerical_heatmap.pkl"
    if os.path.exists(savefile):
        used_args, analytical_data, numerical_data = load_simulation_data(savefile)
    else:
        used_args = fgm_args.copy()
        analytical_data = np.zeros((len(n_values), len(distance_values),time_steps+2))
        numerical_data = np.zeros((len(n_values), len(distance_values),time_steps+2))
        used_args['display_fixation'] = False
        sigma_add = used_args['sigma_add']
        duplication_rate = used_args.get('duplication_rate', 0.02)
        for i, n in enumerate(n_values):
            used_args['n'] = n
            for j, initial_distance in enumerate(distance_values):
                used_args['initial_distance'] = initial_distance

                # Analytical solution
                analytical_distances, _ = analytical_simulation(n, initial_distance, sigma_add, time_steps+2, duplication_rate)

                # Numerical simulations
                numerical_distances = np.zeros((num_runs, time_steps+2))
                for run in range(num_runs):
                    fgm = FisherGeometricModel(**used_args)
                    fgm.evolve_successive(time_steps)
                    numerical_distances[run] = np.linalg.norm(fgm.positions, axis=1)
                    print(f"Run {run + 1}/{num_runs} for n={n}, initial_distance={initial_distance} completed", end="\r")

                # Calculate the average numerical distances
                avg_numerical_distances = np.mean(numerical_distances, axis=0)

                # Compute the integrated square of the difference
                numerical_data[i, j] = avg_numerical_distances
                analytical_data[i,j] = analytical_distances
        used_args["initial_distance"] = distance_values
        used_args["n"] = n_values
        used_args["time_steps"] = time_steps
        save_simulation_data(savefile, (used_args,analytical_data, numerical_data))

    time_steps = used_args["time_steps"]

    heatmap_data = np.zeros((len(n_values), len(distance_values)))
    normalized_heatmap_data = np.zeros((len(n_values), len(distance_values)))
    for i in range(len(n_values)):
        for j in range(len(distance_values)):
            analytical = analytical_data[i,j]
            numerical = numerical_data[i,j]
            heatmap_data[i,j] = np.trapezoid(abs(analytical - numerical)) / time_steps
            normalized_heatmap_data[i,j] = np.trapezoid(abs((analytical - numerical)/analytical)) / time_steps

    # Plotting the heatmap
    fig = plt.figure(figsize=(20, 8))
    ax1, ax2 = fig.subplots(1,2)

    heatmap = sns.heatmap(heatmap_data, ax = ax1, cmap='viridis', cbar=True)
    cbar = heatmap.collections[0].colorbar
    cbar.set_label('Difference', fontsize=20)
    cbar.ax.tick_params(labelsize=16)
    ax1.set_xticks(np.arange(len(distance_values)) + 0.5, rotation = 45)
    ax1.set_xticklabels(distance_values)
    ax1.set_yticks(np.arange(len(n_values)) + 0.5, rotation = 0)
    ax1.set_yticklabels(n_values)
    ax1.invert_yaxis()
    ax1.set_xlabel('Initial Distance to Optimum', fontsize=20)
    ax1.set_ylabel('Dimension n', fontsize=20)
    ax1.text(0.01, 0.99, 'a)', transform=ax1.transAxes, fontsize=20, va='top', bbox=dict(facecolor='0.7', edgecolor='none', pad=3.0))

    
    heatmap = sns.heatmap(normalized_heatmap_data, ax = ax2, cmap='viridis', cbar=True)
    cbar = heatmap.collections[0].colorbar
    cbar.set_label('Normalized Difference', fontsize=20)
    cbar.ax.tick_params(labelsize=16)
    ax2.set_xticks(np.arange(len(distance_values)) + 0.5, rotation = 45)
    ax2.set_xticklabels(distance_values)
    ax2.set_yticks(np.arange(len(n_values)) + 0.5, rotation = 0)
    ax2.set_yticklabels(n_values)
    ax2.invert_yaxis()
    ax2.set_xlabel('Initial Distance to Optimum', fontsize=20)
    ax2.set_ylabel('Dimension n', fontsize=20)
    ax2.text(0.01, 0.99, 'b)', transform=ax2.transAxes, fontsize=20, va='top', bbox=dict(facecolor='0.7', edgecolor='none', pad=3.0))

    print("Used arguments for analytical v numerical solution heatmap plot:", used_args)
    fig.savefig("Figures/analytical_vs_numerical_heatmap")
    return fig

def save_simulation_data(filename, data):
    with open(filename, 'wb') as file:
        pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)

def load_simulation_data(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def show_duplication_events(fgm_args: dict, num_runs: int = 100, time_steps: int = 1000):
    """
    Compare the analytical probability of duplication with actual duplication events during simulation.
    
    Parameters:
    - fgm_args: dict, arguments for initializing the FisherGeometricModel
    - num_runs: int, number of numerical simulations to run
    - time_steps: int, number of time steps for the simulation
    - save_file: str, filename to save/load the simulation data
    
    Returns:
    - fig: matplotlib figure, the 3D histogram plot
    """
    savefile = "test_data/duplication_events.pkl"
    used_args = fgm_args.copy()
    n = used_args['n']
    if os.path.exists(savefile):
        gene_sizes, distances = load_simulation_data(savefile)
    else:
        
        used_args['display_fixation'] = False

        gene_sizes = []
        distances = []

        for run in range(num_runs):
            fgm = FisherGeometricModel(**used_args)
            fgm.evolve_successive(time_steps)
            for event in fgm.mutation_events["duplication"]:
                if event['fixed']:
                    gene_sizes.append(np.linalg.norm(event["gene"]))
                    distances.append(np.linalg.norm([event["pos"]]))

            print(f"Run {run + 1}/{num_runs} completed", end="\r")

        gene_sizes = np.array(gene_sizes)
        distances = np.array(distances)

        save_simulation_data(savefile, (gene_sizes, distances))

    # Create a 3D histogram
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    hist, xedges, yedges = np.histogram2d(gene_sizes, distances, bins=30)

    xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0

    dx = dy = 0.5 * np.ones_like(zpos)
    dz = hist.ravel()

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average', alpha=0.6)

    # # Overlay the analytical probability
    # X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
    # Z = np.zeros_like(X)
    # for i in range(X.shape[0]):
    #     for j in range(X.shape[1]):
    #         Z[i, j] = analytical_probability(X[i, j], Y[i, j], n)

    # ax.plot_surface(X, Y, Z, color='r', alpha=0.5)

    ax.set_xlabel('Gene Size', fontsize=14)
    ax.set_ylabel('Distance to Optimum', fontsize=14)
    ax.set_zlabel('Duplication Events', fontsize=14)

    fig.savefig("Figures/duplication_events")
    return fig

def compare_genome_size_no_duplications(fgm_args: dict, genome_sizes: list[int], num_runs: int = 100, time_steps: int = 1000):
    """
    Compare the evolution of distance to the optimum for different genome sizes without duplications.
    
    Parameters:
    - fgm_args: dict, arguments for initializing the FisherGeometricModel
    - genome_sizes: list[int], list of different initial genome sizes
    - num_runs: int, number of numerical simulations to run for each genome size
    - time_steps: int, number of time steps for the simulation
    
    Returns:
    - fig: matplotlib figure, the comparison plot
    """
    savefile = "test_data/genome_size_no_duplications_comparison.pkl"
    if os.path.exists("test_data/genome_size_no_duplications_comparison.pkl"):
        used_args, results = load_simulation_data(savefile)
    else:
        used_args = fgm_args.copy()
        used_args['mutation_methods'] = ['addition']
        used_args['display_fixation'] = False

        results = {}

        for genome_size in genome_sizes:
            distances = np.zeros((num_runs, time_steps+2))
            for run in range(num_runs):
                fgm = FisherGeometricModel(**used_args)
                fgm.evolve_successive(time_steps)
                distances[run] = np.linalg.norm(fgm.positions, axis=1)
                print(f"Run {run + 1}/{num_runs} for genome size={genome_size} completed", end="\r")
            results[genome_size] = distances
        used_args["genome_sizes"] = genome_sizes
        used_args["num_runs"] = num_runs
        save_simulation_data(savefile, (used_args,results))

    # Plotting the results
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot()

    for genome_size, distances in results.items():
        avg_distances = np.mean(distances, axis=0)
        ax.plot(range(time_steps+2), avg_distances, label=f"Genome size {genome_size}")

    ax.set_xlabel("Time", fontsize=14)
    ax.set_ylabel("Distance to the optimum", fontsize=14)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.legend(fontsize=12)
    ax.grid()
    ax.tick_params(axis='both', which='major', labelsize=12)

    print("Model arguments for genome size comparison plot:", used_args)
    fig.savefig("Figures/genome_size_no_duplications_comparison")
    return fig

def compare_duplication_attempts(fgm_args: dict, num_runs: int = 100, time_steps: int = 1000):
    """
    Compare the proportion of successful duplication attempts from a number of simulations, grouped by the size of r/d.
    
    Parameters:
    - fgm_args: dict, arguments for initializing the FisherGeometricModel
    - num_runs: int, number of numerical simulations to run
    - time_steps: int, number of time steps for the simulation
    - save_file: str, filename to save/load the simulation data
    
    Returns:
    - fig: matplotlib figure, the comparison plot
    """
    savefile = "test_data/duplication_attempts_comparison.pkl"
    used_args = fgm_args.copy()
    n = used_args['n']
    if os.path.exists(savefile):
        analytical_probs, bin_centers, success_proportions = load_simulation_data(savefile)
    else:
        used_args['display_fixation'] = False

        r_d_ratios = []
        success_counts = []
        attempt_counts = []

        for run in range(num_runs):
            fgm = FisherGeometricModel(**used_args)
            fgm.evolve_successive(time_steps)
            for event in fgm.mutation_events["duolication"]:
                r = np.linalg.norm(event["gene"])
                d = np.linalg.norm(event["pos"]-event["gene"])
                r_d_ratio = r / d
                r_d_ratios.append(r_d_ratio)
                attempt_counts.append(1)
                success_counts.append(1 if event['fixed'] else 0)
            print(f"Run {run + 1}/{num_runs} completed", end="\r")

        r_d_ratios = np.array(r_d_ratios)
        success_counts = np.array(success_counts)
        attempt_counts = np.array(attempt_counts)
        
        # Group by r/d ratios
        bins = np.linspace(0, 2/3, 30)
        bin_indices = np.digitize(r_d_ratios, bins)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        success_proportions = np.zeros(len(bins) - 1)

        for i in range(1, len(bins)):
            bin_mask = bin_indices == i
            if np.sum(bin_mask) > 0:
                success_proportions[i - 1] = np.sum(success_counts[bin_mask]) / np.sum(attempt_counts[bin_mask])

        # Analytical probability
        analytical_probs = np.array([analytical_probability(r, 1, n) for r in np.linspace(0, 2/3, 29)])

        save_simulation_data(savefile, (analytical_probs, bin_centers, success_proportions))


    # Plotting the results
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot()

    ax.plot(bin_centers, success_proportions, label="Simulated Proportion of Successful Duplications", color="b")
    ax.plot(bin_centers, analytical_probs, label="Analytical Probability", color="r")

    ax.set_xlabel("r/d Ratio", fontsize=14)
    ax.set_ylabel("Proportion of Successful Duplications", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid()
    ax.tick_params(axis='both', which='major', labelsize=12)


    fig.savefig("Figures/duplication_attempts_comparison")
    return fig

def compare_duplication_heatmap(fgm_args: dict, n_values: list[int], num_runs: int = 100, generations : int = 10000):
    """
    Generate a heatmap showing the proportions of beneficial observed duplications for a range of n values,
    and compare it with the heatmap of analytical expectations.
    
    Parameters:
    - fgm_args: dict, arguments for initializing the FisherGeometricModel
    - n_values: list[int], list of different values for the dimension n
    - num_runs: int, number of numerical simulations to run for each n value
    - time_steps: int, number of time steps for the simulation
    - save_file: str, filename to save/load the simulation data
    
    Returns:
    - fig: matplotlib figure, the heatmap plot
    """
    savefile : str = "test_data/duplication_heatmap_data.pkl"
    used_args = fgm_args.copy()
    if os.path.exists(savefile):
        used_args, r_values, observed_heatmap, analytical_heatmap = load_simulation_data(savefile)
    else:
        used_args['display_fixation'] = False
        used_args['mutation_methods'] = ['addition','duplication']
        used_args['save_unfixed_mutations'] = True
        r_values = np.linspace(0, 2/3, 30)
        observed_heatmap = np.zeros((len(n_values), len(r_values)))
        analytical_heatmap = np.zeros((len(n_values), len(r_values)))

        for i, n in enumerate(n_values):
            r_d_ratios = []
            success_counts = []
            attempt_counts = []

            for run in range(num_runs):
                used_args['n'] = n
                fgm = FisherGeometricModel(**used_args)
                fgm.evolve_successive(generations)
                for event in fgm.mutation_events["duplication"]:
                    r = np.linalg.norm(event["gene"])
                    d = np.linalg.norm(event["pos"]-event["gene"])
                    r_d_ratio = r / d
                    r_d_ratios.append(r_d_ratio)
                    attempt_counts.append(1)
                    success_counts.append(1 if event['fixed'] else 0)
                print(f"Run {run + 1}/{num_runs} for n={n} completed", end="\r")

            r_d_ratios = np.array(r_d_ratios)
            success_counts = np.array(success_counts)
            attempt_counts = np.array(attempt_counts)

            # Group by r/d ratios
            bin_indices = np.digitize(r_d_ratios, r_values)
            for j in range(1, len(r_values)):
                bin_mask = bin_indices == j
                if np.sum(bin_mask) > 0:
                    observed_heatmap[i, j - 1] = np.sum(success_counts[bin_mask]) / np.sum(attempt_counts[bin_mask])
                analytical_heatmap[i, j - 1] = analytical_probability(r_values[j - 1], 1, n)

        save_simulation_data(savefile, (used_args, r_values, observed_heatmap, analytical_heatmap))

    # Calculate the difference heatmap
    difference_heatmap = observed_heatmap - analytical_heatmap

    # Plotting the heatmaps
    fig, axes = plt.subplot_mosaic("AB;CC", figsize=(16, 9))
    ticks = r_values[::5]
                            
    heatmap_A = sns.heatmap(observed_heatmap, ax=axes['A'], cmap='viridis', cbar = True)
    cbar_A = heatmap_A.collections[0].colorbar
    cbar_A.set_label('Observed Proportion', fontsize=14)
    cbar_A.ax.tick_params(labelsize=12)
    axes['A'].set_xticks(np.arange(len(ticks))*5 + 0.5)
    axes['A'].set_xticklabels(np.round(ticks, 2), rotation=45)
    axes['A'].set_yticks(np.arange(len(n_values)) + 0.5)
    axes['A'].set_yticklabels(n_values, rotation = 0)
    axes['A'].invert_yaxis()
    axes['A'].set_xlabel('r/d Values', fontsize=14)
    axes['A'].set_ylabel('n Values', fontsize=14)
    axes['A'].text(0.93, 0.98, 'a)', transform=axes['A'].transAxes, fontsize=16, va='top', bbox=dict(facecolor='0.7', edgecolor='none', pad=3.0))

    heatmab_B = sns.heatmap(analytical_heatmap, ax=axes['B'], cmap='viridis', cbar=True)
    cbar_B = heatmab_B.collections[0].colorbar
    cbar_B.set_label('Predicted Proportion', fontsize=14)
    cbar_B.ax.tick_params(labelsize=12)
    
    axes['B'].set_xticks(np.arange(len(ticks))*5 + 0.5)
    axes['B'].set_xticklabels(np.round(ticks, 2), rotation=45)
    axes['B'].set_yticks(np.arange(len(n_values)) + 0.5)
    axes['B'].set_yticklabels(n_values, rotation = 0)
    axes['B'].invert_yaxis()
    axes['B'].set_xlabel('r/d Values', fontsize=14)
    axes['B'].set_ylabel('n Values', fontsize=14)
    axes['B'].text(0.93, 0.98, 'b)', transform=axes['B'].transAxes, fontsize=16, va='top', bbox=dict(facecolor='0.7', edgecolor='none', pad=3.0))



    heatmap_C = sns.heatmap(difference_heatmap, ax=axes['C'], cmap='coolwarm', center=0, cbar = True, vmin = -1, vmax = 1)
    cbar_C = heatmap_C.collections[0].colorbar
    cbar_C.set_label('Observed - Predicted', fontsize=14)
    cbar_C.ax.tick_params(labelsize=12)
    
    axes['C'].set_xticks(np.arange(len(ticks))*5 + 0.5)
    axes['C'].set_xticklabels(np.round(ticks, 2), rotation=45)
    axes['C'].set_yticks(np.arange(len(n_values)) + 0.5)
    axes['C'].set_yticklabels(n_values, rotation = 0)
    axes['C'].invert_yaxis()
    axes['C'].set_xlabel('r/d Values', fontsize=14)
    axes['C'].set_ylabel('n Values', fontsize=14)
    axes['C'].text(0.93, 0.98, 'c)', transform=axes['C'].transAxes, fontsize=16, va='top', bbox=dict(facecolor='0.7', edgecolor='none', pad=3.0))

    fig.savefig("Figures/duplication_heatmap_comparison")
    print(f"Model arguments for duplication heatmap plot: {used_args}")
    return fig

def observed_deletion_heatmap(fgm_args: dict, n_values: list[int], num_runs: int = 100, generations : int = 10000):
    """
    Generate a heatmap showing the proportions of beneficial observed deletion for a range of n values,
    and compare it with the heatmap of analytical expectations.
    
    Parameters:
    - fgm_args: dict, arguments for initializing the FisherGeometricModel
    - n_values: list[int], list of different values for the dimension n
    - num_runs: int, number of numerical simulations to run for each n value
    - time_steps: int, number of time steps for the simulation
    - save_file: str, filename to save/load the simulation data
    
    Returns:
    - fig: matplotlib figure, the heatmap plot
    """
    savefile : str = "test_data/deletion_heatmap_data.pkl"
    used_args = fgm_args.copy()
    if os.path.exists(savefile):
        used_args, observed_heatmap = load_simulation_data(savefile)
    else:
        used_args['display_fixation'] = False
        used_args['mutation_methods'] = ['addition','duplication', 'deletion']
        used_args["save_unfixed_mutations"] = True

        r_values = np.linspace(0, 1, 30)
        observed_heatmap = np.zeros((len(n_values), len(r_values)))

        for i, n in enumerate(n_values):
            r_d_ratios = []
            success_counts = []
            attempt_counts = []

            for run in range(num_runs):
                used_args['n'] = n
                fgm = FisherGeometricModel(**used_args)
                fgm.evolve_successive(generations)
                for event in fgm.mutation_events["deletion"]:
                    r = np.linalg.norm(event["gene"])
                    d = np.linalg.norm(event["pos"]-event["gene"])
                    r_d_ratio = r / d
                    r_d_ratios.append(r_d_ratio)
                    attempt_counts.append(1)
                    success_counts.append(1 if event['fixed'] else 0)
                print(f"Observed deletions: run {run + 1}/{num_runs} for n={n} completed", end="\r")

            r_d_ratios = np.array(r_d_ratios)
            success_counts = np.array(success_counts)
            attempt_counts = np.array(attempt_counts)

            # Group by r/d ratios
            bin_indices = np.digitize(r_d_ratios, r_values)
            for j in range(1, len(r_values)):
                bin_mask = bin_indices == j
                if np.sum(bin_mask) > 0:
                    observed_heatmap[i, j - 1] = np.sum(success_counts[bin_mask]) / np.sum(attempt_counts[bin_mask])
        used_args["n_values"] = n_values
        used_args["r_values"] = r_values
        save_simulation_data(savefile, (used_args, observed_heatmap))

    n_values = used_args['n_values']
    r_values = used_args['r_values']
    
    # Plotting the heatmap
    fig, ax = plt.subplots(figsize=(16, 9))
    ticks = r_values[::5]

    heatmap_A = sns.heatmap(observed_heatmap, ax=ax, cmap='viridis', cbar = True)
    cbar_A = heatmap_A.collections[0].colorbar
    cbar_A.set_label('Observed Proportion', fontsize=14)
    cbar_A.ax.tick_params(labelsize=12)
    ax.set_xticks(np.arange(len(ticks))*5 + 0.5)
    ax.set_xticklabels(np.round(ticks, 2), rotation=45)
    ax.set_yticks(np.arange(len(n_values)) + 0.5)
    ax.set_yticklabels(n_values, rotation = 0)
    ax.invert_yaxis()
    ax.set_xlabel('r/d Values', fontsize=14)
    ax.set_ylabel('n Values', fontsize=14)

    fig.savefig("Figures/deletion_observed_proportion_heatmap")
    print(f"Model arguments for deletion heatmap plot: {used_args}")
    return fig

def test_mean_simulation_results(fgm_args: dict, n_values: list[int] = [10,50,100], mutation_methods: list[list[str]] = None, num_runs: int = 100, time_steps: int = 1000):
    """
    Test the mean simulation results for the GFGM with different mutation methods and values of n.
    
    Parameters:
    ----
    fgm_args: dict, 
        arguments for initializing the FisherGeometricModel
    n_values: list[int], 
        list of different values for the dimension n
    mutation_methods: list[list[str]], 
        list of different mutation methods to test
    num_runs: int, 
        number of numerical simulations to run for each combination of n and mutation methods
    time_steps: int, 
        number of time steps for the simulation
    
    Returns:
    ----
    fig: matplotlib figure, 
        the comparison plot
    """

    if os.path.exists("test_data/mean_simulation_results.pkl"):
        used_args, all_distances, fixed_mutation_events, all_genome_ranges = load_simulation_data("test_data/mean_simulation_results.pkl")
    else:
        if mutation_methods == None:
            mutation_methods = [
                                ["addition", "duplication", "deletion"],
                                ["addition", "duplication"],
                                ["addition"]]

        used_args = fgm_args.copy()
        used_args["display_fixation"] = False


        all_distances = np.zeros((len(n_values), len(mutation_methods), num_runs, time_steps+2))
        all_genome_ranges = np.zeros((len(n_values), len(mutation_methods), num_runs, time_steps+2))
        fixed_mutation_events = dict()

        for i, n in enumerate(n_values):
            for j, methods in enumerate(mutation_methods):
                used_args['n'] = n
                used_args['mutation_methods'] = methods

                for run in range(num_runs):
                    fgm = FisherGeometricModel(**used_args)
                    fgm.evolve_successive(time_steps)
                    all_distances[i, j, run] = np.linalg.norm(fgm.positions, axis=1)
                    all_genome_ranges[i, j, run] = fgm.total_genome_range
                    fixed_mutation_events[(i, j, run)] = { event_name : [event for event in events if event["fixed"]] for event_name, events in fgm.mutation_events.items()}
                    print(f"Run {run + 1}/{num_runs} for n={n}, methods={methods} completed", end="\033[K\r")


        used_args["num_runs"] = num_runs
        used_args["time_steps"] = time_steps
        used_args["n_values"] = n_values
        used_args["mutation_methods"] = mutation_methods
        save_simulation_data("test_data/mean_simulation_results.pkl", (used_args, all_distances, fixed_mutation_events, all_genome_ranges))

    #Plotting data
    time_steps = used_args["time_steps"]
    x = np.arange(time_steps+2)

    fig, axes = plt.subplot_mosaic("a;a;a;b;c;d;e;e;e",figsize=(6, 11))
    # axes["C"].yaxis.set_ticklabels([])
    # axes["D"].yaxis.set_ticklabels([])

    linetypes = ['-.','--', '-']
    colors = ['g', 'r', 'b']
    percentiles = [20, 80]

    color_handles = [Line2D([0], [0], color=color, lw=2) for color in colors]
    line_handles = [Line2D([0], [0], color='black', lw=2, linestyle=line) for line in linetypes]
    color_labels = ['Point, Duplications and Deletions', 'Point and Duplications', 'Only Point']
    line_labels = used_args["n_values"]
    log_y = True
    log_x = True

    if log_y and log_x:
        location = 'lower left'
        anchor1 = (0,0)
        anchor2 = (0, 0.3)  
    else:
        location = 'upper right'
        anchor1 = (1,1)
        anchor2 = (1, 0.8)
    for i, ax in axes.items():
        if log_x:
            ax.set_xscale("log")
    axes['a'].set_yscale("log")

    def plot_distances(ax : plt.Axes):
        for i, n in enumerate(used_args["n_values"]):
            for j, methods in enumerate(used_args["mutation_methods"]):
                lower = np.percentile(all_distances[i, j], percentiles[0], axis=0)
                upper = np.percentile(all_distances[i, j], percentiles[1], axis=0)
                mean = np.mean(all_distances[i, j], axis=0)
                ax.plot(x, mean, label=f"Mean Distance (n={n}, methods={methods})", linestyle=linetypes[i], c = colors[j])
                ax.fill_between(x, lower, upper, alpha=0.3, color = colors[j])
        
        legend1 = ax.legend(color_handles, color_labels, title="Mutation Type", loc=location, bbox_to_anchor=anchor1, fontsize=9)
        legend2 = ax.legend(line_handles, line_labels, title="Dimension", loc=location, bbox_to_anchor=anchor2, fontsize = 9)

        ax.add_artist(legend1) # Add the customed legend


        # ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Distance to the Optimum", fontsize=12)
        
        ax.grid()
        ax.tick_params(axis='both', which='major', labelsize=10)
        return ax

    def plot_mutation_events(ax : plt.Axes, method : str):
        for i, n in enumerate(used_args["n_values"]):
            for j, methods in enumerate(used_args["mutation_methods"]):
                if method not in methods:
                    continue
                all_fix_times = np.zeros((num_runs, time_steps+2))
                for run in range(num_runs):
                    fixed_mutation_times = [event["time"] for event in fixed_mutation_events[(i, j, run)][method] if event["fixed"]]
                    all_fix_times[run, fixed_mutation_times] = 1
                cumsum = np.cumsum(all_fix_times, axis=1)
            
                upper = np.percentile(cumsum, percentiles[1], axis=0)
                mean = np.mean(cumsum, axis=0)
                lower = np.percentile(cumsum, percentiles[0], axis=0)

                ax.plot(x, mean, linestyle=linetypes[i], c = colors[j])
                # ax.fill_between(x, lower, upper, alpha=0.3, color = colors[j])
        ax.grid()

        return ax
    
    def plot_mean_gene_size(ax : plt.Axes):
        for i, n in enumerate(used_args["n_values"]):
            for j, methods in enumerate(used_args["mutation_methods"]):
                gene_ranges = all_genome_ranges[i, j] / used_args["initial_distance"]

                # lower = np.percentile(gene_sizes, percentiles[0], axis=0)
                # upper = np.percentile(gene_sizes, percentiles[1], axis=0)
                mean = np.mean(gene_ranges, axis = 0)
                ax.plot(x, mean, linestyle=linetypes[i], c = colors[j])
                # ax.fill_between(x, lower, upper, alpha=0.3, color = colors[j])

        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Normalized Gene Size", fontsize=12)
        
        ax.grid()
        ax.tick_params(axis='both', which='major', labelsize=10)
        return ax
    
    
    plot_distances(axes["a"])
    axes["a"].text(0.1, 0.9, 'a)', transform=axes["a"].transAxes, fontsize=16, va='top', bbox=dict(facecolor='0.7', edgecolor='none', pad=3.0))
    for method, i in zip(["addition", "duplication", "deletion"], "bcd"):
        plot_mutation_events(axes[i], method)
        axes[i].text(0.1, 0.9, f'{i})', transform=axes[i].transAxes, fontsize=16, va='top', bbox=dict(facecolor='0.7', edgecolor='none', pad=3.0))
    axes['b'].set_ylabel("Point mutations", fontsize=12)
    axes['c'].set_ylabel("Duplications", fontsize=12)
    axes['d'].set_ylabel("Deletions", fontsize=12)
    plot_mean_gene_size(axes['e'])
    axes['e'].text(0.1, 0.9, 'e)', transform=axes['e'].transAxes, fontsize=16, va='top', bbox=dict(facecolor='0.7', edgecolor='none', pad=3.0))
    
    # plot_mean_gene_size(axes["e"])
    fig.tight_layout()
    fig.savefig("Figures/mean_simulation_results")
    return fig

def find_fixed_mutations(fgm_args: dict, nb_runs: int = 100, time_steps: int = 1000):
    """
    Compare the average number of fixed duplications and deletions from a number of simulations.
    
    Parameters:
    - fgm_args: dict, arguments for initializing the FisherGeometricModel
    - num_runs: int, number of numerical simulations to run
    - time_steps: int, number of time steps for the simulation
    
    Returns:
    - fig: matplotlib figure, the overlapping histograms plot
    """
    savefile = "test_data/fixed_mutations.pkl"
    if os.path.exists(savefile):
        used_args, points, duplications, deletions = load_simulation_data(savefile)
    else:
        used_args = fgm_args.copy()
        used_args['display_fixation'] = False
        used_args['mutation_methods'] = ['addition', 'duplication', 'deletion']

        points = np.array([])
        duplications = np.array([])
        deletions = np.array([])
        
        for run in range(nb_runs):
            fgm = FisherGeometricModel(**used_args)
            fgm.evolve_successive(time_steps)
            
            points = np.concatenate((points, [i for i, method in enumerate(fgm.methods) if method[0]])) 
            duplications = np.concatenate((duplications, [i for i, method in enumerate(fgm.methods) if method[1]]))
            deletions = np.concatenate((deletions, [i for i, method in enumerate(fgm.methods) if method[2]]))
            print(f"Fixation event run {run + 1}/{nb_runs} completed", end="\r")
        
        used_args["nb_runs"] = nb_runs
        used_args["time_steps"] = time_steps
        save_simulation_data(savefile, (used_args, points, duplications, deletions))

    # Plotting the histograms
    time_steps = used_args["time_steps"]
    nb_runs = used_args["nb_runs"]
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot()

    bins = np.arange(time_steps+2, step=50)
    ax.hist(duplications, bins = bins, alpha=0.5, label="Fixed Duplications", color="b")
    ax.hist(deletions, bins = bins, alpha=0.5, label="Fixed Deletions", color="r")
    # ax.hist(points, alpha=0.5, label="Fixed Point", color="g")

    ax.set_xlabel("Time", fontsize=14)
    ax.set_ylabel("Frequency", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid()
    ax.tick_params(axis='both', which='major', labelsize=12)

    print(f"Model arguments for fixed mutations plot: {used_args}")
    fig.savefig("Figures/fixed_mutation_events")
    return fig

def test_test(fgm: FisherGeometricModel):
    time_steps = fgm.current_time
    x = np.arange(time_steps)
    fig, ax = plt.subplots()

    nb_genes = fgm.get_nb_genes()
    fixed_mutation_times = [event["time"] for event in fgm.mutation_events["addition"] if event["fixed"]]

    all_fix_times = np.zeros(time_steps)
    all_fix_times[fixed_mutation_times] = 1
    weighted_fix_times = all_fix_times*nb_genes
    cumsum = np.cumsum(all_fix_times)
    weighted_cumsum = np.cumsum(weighted_fix_times)

    plt.plot(x, cumsum, label = "Fixed Event Generations")
    plt.plot(x, weighted_cumsum, label = "Weighted Fixed Events")
    plt.legend()
    plt.grid()

if __name__ == "__main__":
    #To create new data for the tests, remove the relevant test data files in test_data folder

    with open('test_data/FisherObject', 'rb') as file:
        fgm :FisherGeometricModel = pickle.load(file)

    with open("Parameters.json", 'rb') as file:
        fgm_args : dict = json.load(file)

    # show_simulation_results(fgm)


    # reset_points = [100, 500, 2000]
    # compare_runs_with_reset(fgm_args, reset_points, total_generations=4000, nb_runs=100)

    # test_deletion_probabilities(fgm_args)

    # test_analytical_vs_numerical(fgm_args, num_runs=100, time_steps=10000)
    
    # n_values = [3,10, 20, 50, 100]
    # distance_values = [10, 20, 50, 100]
    # compare_analytical_numerical_heatmap(fgm_args, n_values, distance_values, num_runs=100, time_steps=10000)
    
    # show_duplication_events(fgm_args, num_runs=1000, time_steps=1000)
    
    # genome_sizes = [1, 5, 10, 20]
    # compare_genome_size_no_duplications(fgm_args, genome_sizes, num_runs=100, time_steps=10000)
    
    # compare_duplication_attempts(fgm_args, num_runs=1000, time_steps=1000)
        
    # n_values = [3, 10, 20, 50, 100]
    # compare_duplication_heatmap(fgm_args, n_values, num_runs=500, generations=10000)

    test_mean_simulation_results(fgm_args, n_values = [100,50,10], num_runs=100, time_steps=100000)

    # find_fixed_mutations(fgm_args, nb_runs=100, time_steps=1000)

    # observed_deletion_heatmap(fgm_args, n_values=[3, 10, 20, 50, 100], num_runs=100, generations=10000)
    # test_test(fgm)
    # plt.show()