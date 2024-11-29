from Rewritten_GFGM import FisherGeometricModel
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import colors
import pickle
import json


def historicity_test(fgm : FisherGeometricModel):
    """
    Resumes Evolution after the 10**5 generations to see if there are differences in
    the evolutionnary path taken

    ------
    Parameters : 
        All necessary parameters are fgm defined in the class object (see __init__) : 
        fitness, genes and current_pos
    
    ------
    Return : 
        None 
        Call for evolve_successive again, but with 10**5 less generations 
    
    """
    fgm.fitness = fgm.timestamp_fitness
    fgm.genes = fgm.timestamp_genes
    fgm.current_pos = fgm.timestamp_position
    fgm.current_fitness = fgm.fitness_calc(fgm.current_pos)
    fgm.evolve_successive(3*10**5)

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

def plotting_path(fgm : FisherGeometricModel):
    """
    Plot the path taken by the individual from it's initial phenotype to the optimum 
    in terms of distance to the origin. The abscisse then correspond to the number of 
    fixed mutations.

    ------
    Parameters :
        memory : list
    List of position (vectors) in the phenotypic space after each mutational event.

    ------
    Return
        None
        (show the graph in an other window)
        
    """
    # path = np.linalg.norm(fgm.positions,axis = 1)
    path = fgm.fitness
    fig = plt.figure(figsize = (10,4))
    ax = fig.add_subplot()
    ax.plot(path)
    ax.set_xlabel('Time')
    ax.set_ylabel('Fitness')
    ax.set_title('Evolution of fitness')
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.grid()
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
        print("Unable to plot a visalization graph when dimensions > 3")
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

def plotting_size(fgm : FisherGeometricModel):
    """
    Plot the evolution of the number of genes (due to duplication and deletion) 
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
    ci = np.zeros(fgm.current_time)
    ci[1:] = [1.96*fgm.std_size[k]/np.sqrt(fgm.nb_genes[k]) for k in range(1,fgm.current_time)]
    list_lower = [fgm.mean_size[i] - ci[i] for i in range(fgm.current_time)]
    list_upper = [fgm.mean_size[i] + ci[i] for i in range(fgm.current_time)]

    x = np.arange(0, fgm.current_time) # abscisse for the confidence intervals to be plotted

    fig = plt.figure(figsize= (10,4))

    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(fgm.nb_genes)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Number of genes in the genotype')
    ax1.set_title('Evolution of the Number of genes with Time')
    ax1.set_xscale("log")
    ax1.grid()

    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(fgm.mean_size)
    ax2.fill_between(x, list_lower, list_upper, alpha = .1)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Mean size of genes in the genotype')
    ax2.set_title('Evolution of the size of genes with Time')
    ax2.set_xscale("log")
    ax2.grid()
    
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

def test_modularity(fgm_args : dict, tested_methods : list[list[str]], nb_tests : int = 100, fitness_limit : float = 0.95, with_random : bool = True):
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

    fig = plt.figure(figsize= (10,4))
    ax = plt.subplot()
    ax.set_xlabel("Methods used")
    ax.set_ylabel("Final modlarity")
    ax.set_title(f"Simulations reaching a fitness of {fitness_limit}")
    ax.grid()

    m = len(tested_methods)
    modularities = np.zeros(shape = (m, nb_tests))
    fixation_times = np.zeros(shape = (m, nb_tests))

    try:
        fgm_args.pop("mutation_methods")
    except:
        pass
    
    for i,methods in enumerate(tested_methods):
        for j in range(nb_tests):
            fgm = FisherGeometricModel(mutation_methods=methods, **fgm_args)
            fgm.evolve_until_fitness(fitness_limit = fitness_limit)
            modularities[i,j] = fgm.modularities[fgm.current_time-1]
            fixation_times[i,j] = fgm.current_time
        print(f"Method {methods} done!")
    
    if with_random:
        mods = np.zeros(nb_tests)
        n = fgm_args["n"]
        for i in range(nb_tests):
            #TODO: change the distribution of the number of vectors created to better match actual data
            v = np.random.normal(0,1,(np.random.randint(1,101),n))
            modularity = find_modularity(v)
            mods[i] = modularity
        ax.boxplot(mods,tick_labels=[f"Random sets of {n}-dimensional genes"], positions = [m+1])




    ax.boxplot(modularities.T, tick_labels=["With duplication", "Only addition"], positions=np.arange(1,m+1))
    ax.set_ylim([0,1])
    # print(nb_genes)
    fig.tight_layout()
    return fig


"""
def draw_modularity_plot(fgm : FisherGeometricModel):
    fig = plt.figure(figsize = (10,4))
    ax = plt.subplot()
    
    genes = abs(fgm.genes)
    a = np.max(genes,axis = 0)
    # genes /= a
    ax.hist(genes, bins = 'auto', stacked = True)

    return fig
"""

def draw_modularity_plot(fgm:FisherGeometricModel):
    fig = plt.figure(figsize = (10,4))
    ax = plt.subplot()
    
    y = fgm.modularities
    x = np.arange(fgm.current_time+1)

    ax.plot(x,y,color = 'r', label = "Modularity")
    ax.grid()
    ax.set_xscale('log')
    ax.legend()

    return fig

def show_simulation_results(fgm : FisherGeometricModel):
    # print(f"Number of genes in time: {fgm.nb_genes}")
    # print(fgm.methods)
    print(f"Number of {fgm.get_args()['mutation_methods']} = {np.sum(fgm.methods,axis = 0)}")
    print(f"Number of genes: {np.unique(fgm.nb_genes, return_counts=True)}")
    gene_sizes = np.linalg.norm(fgm.genes, axis = 1)
    # print(f"Size of genes: {np.sort(gene_sizes)}")
    print(f"Inital beneficial directions: {fgm.initial_beneficial_directions}")
    # print(f"Genes : {fgm.genes}")
    # print(f"Initial position: {fgm.init_pos}")
    plotting_size(fgm)
    plotting_path(fgm)
    plot_vizualised_path(fgm)
    # # draw_gene_trait_graph(fgm)
    draw_modularity_plot(fgm)
    return

if __name__ == "__main__":
    with open('FisherObject', 'rb') as input:
        fgm :FisherGeometricModel = pickle.load(input)

    with open("Parameters.json", 'rb') as input:
        args = json.load(input)
    args["display_fixation"] = False
    # test_fixation_time(args, ns = range(2,103,5), nb_tests = 10, fitness_limit = 0.95)
    tested_methods = [["addition","multiplication", "duplication", "deletion"],["addition","duplication", "deletion"]]
    test_modularity(args,tested_methods, nb_tests = 100, fitness_limit= 0.9)

    # show_simulation_results(fgm)

    plt.show()