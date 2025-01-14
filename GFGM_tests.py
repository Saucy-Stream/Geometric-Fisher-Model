from Rewritten_GFGM import FisherGeometricModel
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pickle
import json
import copy
from Test_files.mutation_test import analytical_simulation
from Test_files.duplication_test import analytical_probability
import seaborn as sns
import os

def plot_historic_fitness(fgm : FisherGeometricModel, fitness1, fitness2):
    """
    Plot the two evolutionnary path in terms of the evolution of fitness over time.
    The second path is a resume of evolution at a previous distance and genotype 
    of the first one. 
    
    ------
    Parameters :
        fitness1 : list
        List of fitness taken after each mutational event
        fitness2 : list
        List of fitness taken after each mutational event for the second evolution
        
    ------
    Return :
        None
        Plot the graphic

    """
    plt.figure()

    plt.plot(fitness1, color="blue", label="first simulation")
    plt.plot(fitness2, color="red", label="second simulation")
    plt.xlabel('Mutational Event')
    plt.ylabel('Fitness')
    plt.legend()
    plt.title('Evolution of Fitness Over Time ')

    plt.show()

def plotting_results(fgm : FisherGeometricModel, fitness, effects, time):
    """
    Plot two important graphs that allow us to follow the evolution of fitness other time
    and the different value of the fitness effect s other time.

    ------
    Parameters :
        fitness : list
        List of the fitness (float) of the individual after each mutational event
        effects : list
        List of the fitness effect (float) of the modification of the phenotype/genotype 
        that happend at each mutational event.
        time : int
        Number of time step used in the simulation

    ------
    Return
        None
        (show the graphs in an other window)
        
    """
    fig = plt.figure()

    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(fitness)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Fitness')
    ax1.set_title('Evolution of Fitness Over Time')

    ax2 = plt.subplot(1, 2, 2)
    effects = np.array(effects)
    effects = effects[effects>=0] # comment this lign if you also want to see deleterious effect
    ax2.plot(effects, '.', markersize=3)
    ax2.hlines(0, 0 , len(effects), "k", "--")
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Fitness Effect (s)')
    ax2.set_title('Fitness Effects of Mutations Over Time')

    return fig

def plot_vizualised_path(fgm : FisherGeometricModel) -> None:
    """
    Plots the final phenotype and underlying genotype of a 2- or 3-dimensional FisherGeometricModel
    
    Parameters
    -----
    fgm
    
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
    ax.legend()
    return fig

def draw_gene_size(fgms : list[FisherGeometricModel], figtitle = "Size and number of genes from simulations"):
    """
    Plot the evolution of the number & size of genes (due to duplication and deletion) 
    with respect to time.
    
    Parameters
    ------
    nb_genes : list
        List of the number of genes in the genome of the individual at each mutational event (3 per time step)
    mean_size : list
        List of the mean size of genes in the genome of the individual at each iteration
    std_size : list
        List of the standard deviation of the size of genes in the genome of the individual at each iteration
    
    Return
    ------
        None
        (show the graph in an other window)
        
    """
    fig = plt.figure(figsize= (10,4))
    fig.suptitle(figtitle)

    ax1 = plt.subplot(1, 2, 1)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Genes')
    ax1.set_title('Number of genes')
    ax1.set_xscale("log")
    ax1.grid()

    ax2 = plt.subplot(1, 2, 2)

    ax2.set_xlabel('Time')
    ax2.set_ylabel('Mean size')
    ax2.set_title('Size of genes')
    ax2.set_xscale("log")
    ax2.grid()
    
    for i,fgm in enumerate(fgms):
        ax1.plot(fgm.nb_genes, label = f"Run {i+1}")
        ax2.plot(fgm.mean_size, label = f"Run {i+1}")
        ci = np.zeros(fgm.current_time)
        ci[1:] = [1.96*fgm.std_size[k]/np.sqrt(fgm.nb_genes[k]) for k in range(1,fgm.current_time)]
        list_lower = [fgm.mean_size[i] - ci[i] for i in range(fgm.current_time)]
        list_upper = [fgm.mean_size[i] + ci[i] for i in range(fgm.current_time)]

        x = np.arange(0, fgm.current_time) # abscisse for the confidence intervals to be plotted
        ax2.fill_between(x, list_lower, list_upper, alpha = .1)
    ax1.legend()
    ax2.legend()
    return fig

def draw_gene_trait_graph(fgm : FisherGeometricModel) -> None:
    fig = plt.figure(figsize = (10,4))
    ax = plt.subplot()

    graph = fgm.get_graph()
    partition = nx.community.louvain_communities(graph)

    nb_genes = len([node for node in graph.nodes if node[0] == 'g'])
    nb_traits = len([node for node in graph.nodes if node[0] == 't'])
    trait_factor = (nb_genes-1)/(nb_traits-1)

    genes_ordered = np.array([])
    traits_ordered = np.array([])
    community_divider_genes = 0
    community_divider_traits = 0
    for community in partition:
        genes = np.array([node for node in community if node[0] == 'g'])
        traits = np.array([node for node in community if node[0] == 't'])
        genes_ordered = np.concatenate((genes_ordered,genes))
        traits_ordered = np.concatenate((traits_ordered,traits))

        #plot divisor lines between communities
        community_divider_genes += len(genes)
        community_divider_traits += len(traits)
        ax.plot((0,10),(community_divider_genes-0.5,(community_divider_traits-0.5)*trait_factor),c = 'k')


    gene_x_positions = np.full(nb_genes, 0)
    gene_y_positions = range(nb_genes)
    trait_x_positions = np.full(nb_traits, 10)
    trait_y_positions = np.array(range(nb_traits))*trait_factor

    weights = np.array([graph.get_edge_data(*edge)['weight'] for edge in graph.edges])
    largest_weight = np.max(abs(weights))
    for i, gene in enumerate(genes_ordered):
        for j,trait in enumerate(traits_ordered):
            weight = graph.get_edge_data(gene,trait)['weight']
            alpha = abs(weight)/largest_weight

            trait_index = int(trait.split()[-1])
            gene_contributes_to_trait = weight*fgm.init_pos[trait_index] < 0
            if gene_contributes_to_trait:
                color = 'g'
            else:
                color = 'r'
            if alpha >= 0:
                ax.plot((0,10),(i,j*(nb_genes-1)/(nb_traits-1)),c = color,alpha = alpha)
    
    ax.scatter(gene_x_positions,gene_y_positions)
    ax.scatter(trait_x_positions,trait_y_positions)

    return fig

def test_fixation_time(fgm_args : dict, ns: list[int], fitness_limit : float= 0.99, nb_tests : int = 10) -> None:
    fgm_args["display_fixation"] = False
    try:
        fgm_args.pop('n')
    except:
        pass


    nb_genes = np.zeros(shape = (len(ns), nb_tests))
    fixation_times = np.zeros(shape = (len(ns), nb_tests))

    for i,n in enumerate(ns):
        for j in range(nb_tests):
            fgm = FisherGeometricModel(n, **fgm_args)
            fgm.evolve_until_fitness(fitness_limit = fitness_limit)
            nb_genes[i,j] = len(fgm.genes)
            fixation_times[i,j] = fgm.current_time
        print(f"Dimension {n} done!")
    # print(f"Average number of communities for dimensions {ns}: {nb_communities}")

    fig = plt.figure(figsize= (10,4))
    ax = plt.subplot()
    ax.set_xlabel("Number of dimensions")
    ax.set_ylabel("Time until fixation")
    ax.set_title(f"Simulations reaching a fitness of {fitness_limit}")
    # ax.set_yscale('log')
    ax.grid()


    X = np.tile(ns, (nb_tests,1))

    cmap = plt.cm.viridis
    scatter = ax.scatter(X.flatten(), fixation_times.flatten(), c = nb_genes.flatten(), cmap = cmap, linewidths= 5)
    cbar = fig.colorbar(scatter, ax=ax, label='Number of genes at fixation')
    
    # print(nb_genes)
    return fig

def test_methods(fgm_args : dict, tested_methods : list[list[str]], nb_tests : int = 100, fitness_limit : float = 0.95, with_random : bool = True):
    def find_modularity(vecs: np.ndarray[np.ndarray[float]]) -> float:
        vec_copies = np.copy(vecs)
        n = len(vec_copies[0])  # Dimension of the vectors
        scale = 1 / np.sqrt(n)
        weighted_modularity_sum = 0
        total_weight = 0
        for vec in vec_copies:
            vec /= np.linalg.norm(vec)  # Normalize the vec vector
            
            # Calculate the modularity of the vec
            strongest_dir = max(abs(vec))  # Max absolute value of the components
            modularity = (strongest_dir - scale) / (1 - scale)

            # Calculate the weight of the vec based on its magnitude
            weight = np.linalg.norm(vec)  # Weight is the norm (magnitude) of the vector

            # Add to the weighted sum of modularity and the total weight
            weighted_modularity_sum += modularity * weight
            total_weight += weight

        # Return the weighted average modularity
        M = weighted_modularity_sum / total_weight if total_weight != 0 else 0
        return M
    def new_data():
        for i,methods in enumerate(tested_methods):
            for j in range(nb_tests):
                fgm = FisherGeometricModel(mutation_methods=methods, **fgm_args)
                fgm.evolve_until_fitness(fitness_limit = fitness_limit)
                modularities[i,j] = fgm.modularities[fgm.current_time-1]
                fixation_times[i,j] = fgm.current_time
                number_of_genes[i,j] = fgm.nb_genes[fgm.current_time-1]
                print(f"\rMethod {methods} {100*(j+1)/nb_tests :.0f}% done!", end = '')
            print()
        with open("test_object", "wb") as file:
            pickle.dump([modularities,fixation_times,number_of_genes],file,pickle.HIGHEST_PROTOCOL)
        return
    
    fgm_args["display_fixation"] = False
    try:
        fgm_args.pop("mutation_methods")
    except:
        pass
    
    m = len(tested_methods)
    modularities = np.zeros(shape = (m, nb_tests))
    fixation_times = np.zeros(shape = (m, nb_tests))
    number_of_genes = np.zeros(shape = (m, nb_tests))

    renew_data = input("Renew data for test of different mutational methods? (leave empty if no): ")
    if renew_data:
        new_data()

    with open("test_object", 'rb') as file:
        modularities, fixation_times, number_of_genes = pickle.load(file)
    
    fig1 = plt.figure(figsize= (10,4))
    ax1 = fig1.subplots()
    ax1.boxplot(modularities.T, tick_labels=["Angular","Multiplication + addition", "Only addition"], positions=np.arange(1,m+1))
    ax1.set_ylim([-0.1,1.1])
    ax1.set_xlabel("Methods used")
    ax1.set_ylabel("Final modlarity")
    ax1.set_title(f"Simulations reaching a fitness of {fitness_limit}")
    ax1.grid()
    

    fig2 = plt.figure(figsize= (10,4))
    ax2 = fig2.subplots()
    ax2.set_xlabel("Number of genes")
    ax2.set_title(f"Distribution of number of genes at fitness {fitness_limit}")
    ax2.grid()
    ax2.hist(number_of_genes.T,bins = "auto", label = ["Angular","Multiplication + addition", "Only addition"])
    ax2.legend()

    fig1.tight_layout()
    fig2.tight_layout()

    if with_random:
        mods = np.zeros(nb_tests)
        n = fgm_args["n"]
        for i,nb_genes in enumerate(number_of_genes[0]):
            #TODO: change the distribution of the number of vectors created to better match actual data
            v = np.random.normal(0,1,(int(nb_genes),n))
            modularity = find_modularity(v)
            mods[i] = modularity
        ax1.boxplot(mods,tick_labels=[f"Random sets of {n}-dimensional genes"], positions = [m+1])

    fig1.savefig("Figures/modularity_of_different_mutation_types")
    fig2.savefig("Figures/distribution_amount_of_genes")

    
    return 


def draw_phenotype_size(fgm : FisherGeometricModel):
    fig = plt.figure(figsize = (10,4))
    ax = plt.subplot()
    
    genes = abs(fgm.genes)
    a = np.max(genes,axis = 0)
    # genes /= a
    ax.hist(genes, bins = 'auto', stacked = True)

    return fig


def draw_distance_plot(fgms: list[FisherGeometricModel], with_analytical = False, title :str = "Distance to the optimum"):
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

    ax.set_ylabel("Distance")
    ax.set_xlabel("Time")
    ax.set_title(title)
    ax.set_yscale('log')
    ax.set_xscale("log")
    ax.grid()
    ax.legend()
    fig.savefig("Figures/distance_to_optimum")
    return fig

def test_deletion_probabilities(fgm_args : dict, nb_tests : int = 100, fitness_limit : float = 0.9):
    if input("Do you want to generate new data for the deletion tests? (leave empty if no)"):
        fgm_args["display_fixation"] = False
        deletions = np.zeros(nb_tests)
        deletion_index = fgm_args["mutation_methods"].index("deletion")
        for i in range(nb_tests):
            fgm = FisherGeometricModel(**fgm_args)
            fgm.evolve_until_fitness(fitness_limit)
            deletions[i] = sum(fgm.methods[:,deletion_index])
            print(f"\rDeletion tests {100*(i+1)/nb_tests :.0f}% done!", end = '')
        with open("deletion_data", "wb") as file:
            pickle.dump(deletions,file,pickle.HIGHEST_PROTOCOL)
    else:
        with open("deletion_data", "rb") as file:
            deletions = pickle.load(file)
    
    fig = plt.figure(figsize= (10,4))
    ax = fig.subplots()
    # ax.grid()
    ax.set_xlabel("Deletions")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Deletion fixation for {nb_tests} simulations in n = {fgm_args['n']} with starting distance = {fgm_args['initial_distance']}")
    ax.hist(deletions, color = 'r', ec='k')
    fig.savefig(f"Figures/deletion_histogram_n_{fgm_args['n']}")

def test_effect_of_genome_size(fgm_args : dict, generations_until_reset : int, following_generations : int):
    generations_until_reset = int(generations_until_reset)
    following_generations = int(following_generations)

    fgm = FisherGeometricModel(**fgm_args)
    fgm.evolve_successive(generations_until_reset)
    fgm_2 = copy.deepcopy(fgm)
    fgm_2.reinitialize()

    fgm.evolve_successive(following_generations)
    fgm_2.evolve_successive(following_generations)

    fgms = [fgm, fgm_2]
    fig1 = draw_distance_plot(fgms, title = f"Distance to the optimum with n = {fgm_args['n']} and a reset at t={generations_until_reset}")
    fig1.savefig(f"Figures/reset_t_{generations_until_reset}")

    fig2 = draw_gene_size(fgms, figtitle= f"Number and size of genes with n = {fgm_args['n']} and a reset at t={generations_until_reset}")
    fig2.savefig(f"Figures/gene_size_t_{generations_until_reset}")


def show_simulation_results(fgm : FisherGeometricModel):
    # print(f"Number of genes in time: {fgm.nb_genes}")
    # print(fgm.methods)
    print(f"Number of {fgm.get_args()['mutation_methods']} = {np.sum(fgm.methods,axis = 0)}")
    print(f"Number of genes: {np.unique(fgm.nb_genes, return_counts=True)}")
    # gene_sizes = np.linalg.norm(fgm.genes, axis = 1)
    # print(f"Size of genes: {np.sort(gene_sizes)}")
    print(f"Inital beneficial directions: {fgm.initial_beneficial_directions}")
    # print(f"Genes : {fgm.genes}")
    # print(f"Initial position: {fgm.init_pos}")
    draw_gene_size([fgm])
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
    n = fgm_args['n']
    fgm_args['display_fixation'] = False
    initial_distance = fgm_args['initial_distance']
    sigma_add = fgm_args['sigma_add']
    duplication_rate = fgm_args.get('duplication_rate', 0.02)
    with_duplication = "duplication" in fgm_args['mutation_methods']

    # Analytical solution
    analytical_distances = analytical_simulation(n, initial_distance, sigma_add, time_steps, with_duplication, duplication_rate)

    # Numerical simulations
    numerical_distances = np.zeros((num_runs, time_steps+2))
    for run in range(num_runs):
        fgm = FisherGeometricModel(**fgm_args)
        fgm.evolve_successive(time_steps)
        numerical_distances[run] = np.linalg.norm(fgm.positions, axis=1)
        print(f"Run {run + 1}/{num_runs} completed", end="\r")

    # Plotting the results
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot()

    # Plot analytical solution
    ax.plot(range(time_steps), analytical_distances, label="Analytical solution", color="k")

    # Plot numerical simulations
    for run in range(num_runs):
        ax.plot(range(time_steps+2), numerical_distances[run], color="r", alpha=0.1)
    ax.plot(range(time_steps+2), np.mean(numerical_distances, axis=0), label="Average of Numerical simulations", color="r")
    
    ax.set_xlabel("Time")
    ax.set_ylabel("Distance to the optimum")
    # ax.set_yscale("log")
    # ax.set_xscale("log")
    ax.set_title("Comparison of Analytical Solution and Numerical Simulations")
    ax.legend()
    ax.grid()

    fig.savefig("Figures/analytical_vs_numerical_comparison")
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
    heatmap_data = np.zeros((len(n_values), len(distance_values)))

    for i, n in enumerate(n_values):
        for j, initial_distance in enumerate(distance_values):
            fgm_args['n'] = n
            fgm_args['initial_distance'] = initial_distance
            fgm_args['display_fixation'] = False
            sigma_add = fgm_args['sigma_add']
            duplication_rate = fgm_args.get('duplication_rate', 0.02)
            with_duplication = "duplication" in fgm_args['mutation_methods']

            # Analytical solution
            analytical_distances = analytical_simulation(n, initial_distance, sigma_add, time_steps, with_duplication, duplication_rate)

            # Numerical simulations
            numerical_distances = np.zeros((num_runs, time_steps+2))
            for run in range(num_runs):
                fgm = FisherGeometricModel(**fgm_args)
                fgm.evolve_successive(time_steps)
                numerical_distances[run] = np.linalg.norm(fgm.positions, axis=1)
                print(f"Run {run + 1}/{num_runs} for n={n}, initial_distance={initial_distance} completed", end="\r")

            # Calculate the average numerical distances
            avg_numerical_distances = np.mean(numerical_distances, axis=0)

            # Compute the integrated square of the difference
            diff_squared = np.sum((analytical_distances - avg_numerical_distances[:time_steps])**2)
            normalized_diff = diff_squared / np.sum(avg_numerical_distances[:time_steps]**2)
            heatmap_data[i, j] = normalized_diff

    # Plotting the heatmap
    fig = plt.figure(figsize=(10, 8))
    ax = sns.heatmap(heatmap_data, cmap='viridis', cbar_kws={'label': 'Normalized Square Difference'})

    ax.set_xticks(np.arange(len(distance_values)) + 0.5)
    ax.set_xticklabels(distance_values)
    ax.set_yticks(np.arange(len(n_values)) + 0.5)
    ax.set_yticklabels(n_values)
    ax.invert_yaxis()
    ax.set_xlabel('Initial Distance to Optimum')
    ax.set_ylabel('Dimension n')
    ax.set_title('Comparison of Analytical and Numerical Solutions')

    fig.savefig("Figures/analytical_vs_numerical_heatmap")
    return fig

def save_simulation_data(filename, data):
    with open(filename, 'wb') as file:
        pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)

def load_simulation_data(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def compare_duplication_probability(fgm_args: dict, num_runs: int = 100, time_steps: int = 1000, save_file: str = "duplication_data.pkl"):
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
    n = fgm_args['n']
    if os.path.exists(save_file):
        gene_sizes, distances = load_simulation_data(save_file)
    else:
        
        fgm_args['display_fixation'] = False
        initial_distance = fgm_args['initial_distance']
        sigma_add = fgm_args['sigma_add']
        duplication_rate = fgm_args.get('duplication_rate', 0)

        gene_sizes = []
        distances = []

        for run in range(num_runs):
            fgm = FisherGeometricModel(**fgm_args)
            fgm.evolve_successive(time_steps)
            for event in fgm.duplication_events:
                if event['fixed']:
                    gene_sizes.append(np.linalg.norm(event["gene"]))
                    distances.append(np.linalg.norm([event["distance"]]))

            print(f"Run {run + 1}/{num_runs} completed", end="\r")

        gene_sizes = np.array(gene_sizes)
        distances = np.array(distances)

        save_simulation_data(save_file, (gene_sizes, distances))

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

    ax.set_xlabel('Gene Size')
    ax.set_ylabel('Distance to Optimum')
    ax.set_zlabel('Duplication Events')
    ax.set_title(f'Observed Fixed Duplication Events for {num_runs} Simulations')

    fig.savefig("Figures/duplication_probability_comparison")
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
    fgm_args['mutation_methods'] = [method for method in fgm_args['mutation_methods'] if method != 'duplication']
    fgm_args['display_fixation'] = False

    results = {}

    for genome_size in genome_sizes:
        distances = np.zeros((num_runs, time_steps+2))
        for run in range(num_runs):
            fgm = FisherGeometricModel(**fgm_args)
            fgm.evolve_successive(time_steps)
            distances[run] = np.linalg.norm(fgm.positions, axis=1)
            print(f"Run {run + 1}/{num_runs} for genome size={genome_size} completed", end="\r")
        results[genome_size] = distances

    # Plotting the results
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot()

    for genome_size, distances in results.items():
        avg_distances = np.mean(distances, axis=0)
        ax.plot(range(time_steps+2), avg_distances, label=f"Genome size {genome_size}")

    ax.set_xlabel("Time")
    ax.set_ylabel("Distance to the optimum")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_title("Evolution of Distance to the Optimum for Different Genome Sizes (No Duplications)")
    ax.legend()
    ax.grid()

    fig.savefig("Figures/genome_size_no_duplications_comparison")
    return fig

if __name__ == "__main__":
    with open('FisherObject', 'rb') as file:
        fgm :FisherGeometricModel = pickle.load(file)

    with open("Parameters.json", 'rb') as file:
        fgm_args : dict = json.load(file)
    # test_fixation_time(args, ns = range(2,103,5), nb_tests = 10, fitness_limit = 0.95)
    tested_methods = [["addition"],
                      ["addition","duplication"],
                      ["addition","duplication", "deletion"]]
    # test_methods(fgm_args,tested_methods, nb_tests = 100, fitness_limit= 0.9)
    # show_simulation_results(fgm)


    # test_effect_of_genome_size(fgm_args,generations_until_reset = 5e2,following_generations = 1e3)

    # test_deletion_probabilities(fgm_args)

    # test_analytical_vs_numerical(fgm_args, num_runs=100, time_steps=1000)
    # n_values = [3,10, 20, 50, 100, 500]
    # distance_values = [10, 20, 30, 40, 50, 100]
    # compare_analytical_numerical_heatmap(fgm_args, n_values, distance_values, num_runs=100, time_steps=1000)
    # compare_duplication_probability(fgm_args, num_runs=1000, time_steps=1000)
    genome_sizes = [1, 5, 10, 20]
    compare_genome_size_no_duplications(fgm_args, genome_sizes, num_runs=100, time_steps=10000)
    plt.show()