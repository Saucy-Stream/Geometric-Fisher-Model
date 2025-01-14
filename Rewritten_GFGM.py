'''
Genotypic Fisher Geometric Model of Adaptation, 
Iterative version.
'''

##########################  
#### Import Libraries ####
##########################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.lines import Line2D
from multiprocessing import Pool
import networkx as nx
import pickle
import json


##############################
#### Functions definition ####
##############################

class FisherGeometricModel() :
    def __init__(self, 
                 n: int = 3, initial_distance: float = 20, 
                 mutation_methods : list[str] = ["multiplication","addition", "duplication", "deletion"], 
                 alpha: float = 0.5, Q: float = 2, sigma_add: float = 1, duplication_rate: float = 0.01, 
                 deletion_rate: float = 0.01, ratio: float = 1, initial_gene_method: str = "random", 
                 timestamp : float = 1e5, sigma_mult : float = 0.1, display_fixation : bool = False, 
                 reset_rate : float = 0.01, reset_size : float = 1e-2) :
        """
        Initialized the main parameters and variables used in our modified FGM.
        
        Parameters
        ------
        n 
            Dimension of the phenotypic space = number of main indepedent phenotypic traits that are used as axis is the model
        intial_position
            Initial phenotype/genotype = position in the space of the individual.
        alpha
            Robustness parameter\\
            alpha and Q influence the decay rate and the curvature of the fitness function (Tenaillon 2014)
        Q
            Epistasis parameter;\\
            alpha and Q influence the decay rate and the curvature of the fitness function (Tenaillon 2014)
        ratio
            Ratio between sigma_gene and sigma_add (size of the first gene). Affects the rate of duplication versus mutation.
        sigma_add
            Standard deviation of the mutational vector on each axis
        duplication_rate
            Rate of duplication in nb/gene/generation
        deletion_rate
            Rate of deletion in nb/gene/generation 
        initial_gene_method
            Method of choosing the initial gene
        timestamp
            Time to save the process
        
        Return
        ------ 
            None
            (memorize paramters and variables for later use in other methods)  
        """
        self.dimension : int = n
        self.optimum : np.ndarray[float] = np.zeros(n) #define the origin as the fitness optimum
        self.alpha : float = alpha
        self.Q : float = Q 
        self.sigma_mult : float = sigma_mult
        self.sigma_add : float = sigma_add

        self.init_pos : np.ndarray[float] = self.create_initial_starting_position(initial_distance)
        self.genes : np.ndarray[np.ndarray[float]]= np.array([self.create_fixed_first_gene(initial_gene_method)]) # chose the direction/size of the first gene 

        self.current_pos = self.init_pos + np.sum(self.genes, axis = 0) # the real phenotypic position of the individual is computed by adding the genes vectors to the initial position
        self.current_fitness = self.fitness_calc(self.current_pos) # Compute the fitness of the actual phenotype before any mutation happen

        self.duplication_rate : float = duplication_rate 
        self.deletion_rate : float = deletion_rate
        self.reset_rate : float = reset_rate

        self.reset_size = reset_size

        self.mutation_functions = np.array([None]*len(mutation_methods))
        for i,method in enumerate(mutation_methods):
            if method == "multiplication":
                self.mutation_functions[i] = self.multiplicative_point_mutation
            elif method == "addition":
                self.mutation_functions[i] = self.addative_point_mutation
            elif method == "duplication":
                self.mutation_functions[i] = self.duplication
            elif method == "deletion":
                self.mutation_functions[i] = self.deletion
            elif method == "reset":
                self.mutation_functions[i] = self.reset_mutation
            elif method == "angular":
                self.mutation_functions[i] = self.angular_point_mutation
            else:
                raise Warning(f"Unknown mutation method \"{method}\"")


        fit_1 = self.fitness_calc(self.init_pos)
        fit_2 = self.fitness_calc(self.current_pos)
        self.positions : np.ndarray[np.ndarray[float]] = [self.init_pos, self.current_pos] # memorize the to first position, which are the initial phenotype of the individual and its phenotype after addition of the first gene
        self.fitness : np.ndarray[float] = [fit_1, fit_2] # fitness values of the phenotype at each iteration
        self.methods = np.array([[False]*len(mutation_methods),[False]*len(mutation_methods)]) # method use to modificate the genotype at each iteration
        self.modularities = np.array([0,self.find_modularity(self.genes)])
        self.nb_genes = [0,1] # gene count at each generation
        self.mean_size = np.array([0,np.linalg.norm(self.genes[0])]) # mean size of the genes at each iteration
        self.std_size = np.array([0,0])
        self.duplication_events : list[dict]= []

        #For saving a timestamp of the process
        self.timestamp = timestamp
        self.timestamp_fitness = np.array([]) # to memorize the list of fitness after 10**5 generations and resumes evolution at this step
        self.timestamp_position = np.zeros(n) # to memorize the position after 10**5 generations and resumes evolution at this step
        self.timestamp_genes = np.array([]) # to memorize the list of genes after 10**5 generations and resumes evolution at this step
        
        self.current_time = 1
        initial_signs = self.init_pos*self.genes[0]
        self.initial_beneficial_directions = np.sum([1 for i in range(n) if initial_signs[i] < 0])
        self.display_fixation = display_fixation

        self.args = {'n': n, 'initial_distance': initial_distance, 'mutation_methods': mutation_methods, 'alpha' : alpha, 'Q' : Q,  'duplication_rate': duplication_rate, 'deletion_rate': deletion_rate, 'initial_gene_method': initial_gene_method, 'timestamp': timestamp, 'sigma_mult': sigma_mult, 'sigma_add' : sigma_add, 'display_fixation':display_fixation}
    
    def get_args(self):
        return self.args.copy()

    def create_initial_starting_position(self, distance : float) -> np.ndarray[float]:
        initial_position = np.random.normal(0, 1, self.dimension)
        initial_position /= np.linalg.norm(initial_position)
        initial_position *= distance
        return initial_position

    def create_fixed_first_gene(self, method: str):
        """
        Search for a first gene of a fixed size in the direction indicated as parameter. 
        The gene can be parallel to the optimal axe, orthogonal to this same axe, nearly parallel to it, are neutral.
        Use the gradient of the fitness function to know the direction of the optimal exe linking the initial position to the 
        optimum of fitness. Then, project the gene vector on the wanted direction compared to this axe. 
        
        Parameters
        ------
        r
            Parameter which allow to play on the ratio of the standart deviation of the size of a gene and a mutation 
        direction
            Indicate the direction that should have the gene's vector
            Other parameters arre self defined in the class object (see __init__). Useful parameters are init_pos, sigma_add,
            dimension.

        ------
        Return : 
            gene : np.darray
            1-dimensional numpy array of size n (dimension of the phenotypic space) representing the gene.
 
        """

        if method == "random" :
            s = True
            while s: # we consider that the first gene must be at least neutral or beneficial so that the organism survive
                gene = np.random.normal(0, self.sigma_add, self.dimension) # We draw the gene as if it was a mutation of the starting point in the Standart FGM
                new_pos = self.init_pos + gene # compute the new phenotypic position after adding the gene 
                s = np.linalg.norm(self.init_pos) < np.linalg.norm(new_pos) # compute the fitness effect of such a gene.
            # print(np.linalg.norm(gene))
            
        
        elif method == "parallel" : 
            gene = -self.init_pos / np.linalg.norm(self.init_pos) * np.random.normal(0,self.sigma_add) # gene on the axe to the optimum             
            if np.linalg.norm(self.init_pos) < np.linalg.norm(self.init_pos + gene):
                gene = -gene
        return gene

    
    def addative_point_mutation(self, current_genes : np.ndarray[np.ndarray[float]]):
        """
        Randomly mutate every genes in the genotype by adding a gaussian noise to them. The mutation
        may differ from one gene to another. We consider in this case that the mutation rate is per genome
        and have a value of 1, meaning that each gene in the genome gets mutated exactly once per generation.

        Parameters
        ------
        current_genes : np.ndarray[np.ndarray[float]]
            List of genes to be duplicated
        The useful parameters here are dimension, sigma_add

        Return :
        ------ 
        list_genes : np.ndarray[np.ndarray[float]]
            List of 1-dimensional numpy array of size n (dimension of the phenotypic space) representing the genes
            after mutation.
        mut : boolean
            Always put at true in this version because there is 1 mutation per generation (iteration)

        """
        new_genes = current_genes.copy()
        n = len(new_genes)
        m = np.random.normal(0, self.sigma_add, size=(n, self.dimension)) # draw the mutation from a normal distribution of variance n*sigma_add**2 (variance of a sum of mutation)
        
        new_genes = np.array([new_genes[i] + m[i] for i in range(n)]) # modify every genes in the list by adding the corresponding mutation. All genes do not mutate the same way
        # print(f"1st gene: {self.genes[0]}, mutation: {m[0]}")
        mut = True
        return new_genes, mut
   
    def duplication(self, current_genes : np.ndarray[np.ndarray[float]]):
        """
        Duplicate genes from the list of vectors of genes.
        The number of duplication to do is drawn from a poisson distribution having the duplication rate as parameter
        
        Parameters
        ------
        current_genes : np.ndarray[np.ndarray[float]]
            List of genes to be duplicated
        The useful paramaters here are genes, duplication_rate.

        ------
        Return
        list_genes : list
            list of 1-D numpy arrays (genes) modified after duplication of some genes.
        nb_dupl > 0 : boolean
            True if any duplications have been made.

        """
        n = len(current_genes)
        list_genes = np.array(current_genes.copy()) # make a copy of the initial list of genes so that it is not changed if the modification are not fixed afterward
        duplication_occurred = np.random.rand() < self.duplication_rate*n # number of duplication to do. the rate is multiply by the number of gene and the size of the population so that it represent the number of duplication per generation.

        if duplication_occurred:
            added_gene_index = np.random.choice(range(n))
            added_gene = list_genes[added_gene_index]
            list_genes = np.concatenate((list_genes,[added_gene]))
            self.duplication_events.append({'time': self.current_time, 'gene': added_gene, 'distance': self.current_pos, 'fixed': False})
            
            # print(f"nr dupl:{nb_dupl}, indices: {added_gene_index}, added genes : {added_genes}, total list : {list_genes}")
        return list_genes, duplication_occurred
    
    def deletion(self, current_genes : np.ndarray[np.ndarray[float]]):
        """
        Delete genes from the list of vectors of genes.
        The number of deletion to do is drawn from a poisson distribution having the deletion rate as parameter

        Parameters
        ------
        current_genes : np.ndarray[np.ndarray[float]]
            List of genes to be duplicated
        The useful paramaters here are genes, N, deletion_rate.

        Return
        ------        
            list_genes : list
            list of 1-D numpy arrays (genes) modified after deletion of some genes.
            nb_del > 0 : boolean
            True if a deletion have been made.

        """
        n = len(current_genes)
        list_genes = np.array(current_genes.copy()) # make a copy of the initial list of genes so that it is not changed if the modification are not fixed afterward
        deletion_occurred = np.random.rand() < self.deletion_rate*n  # number of duplication to do. the rate is multiply by the number of gene and the size of the population so that it represent the number of duplication per generation.

        if deletion_occurred:
            removed_gene = np.random.choice(range(n))
            list_genes = np.delete(list_genes,removed_gene,0)

        return list_genes, deletion_occurred

    def fitness_calc(self, position) -> float: # 4sec
        """
        Compute the fitness of a point, depending on its position in the phenotypic space.

        ------
        Parameters :
            position : np.ndarray
            1-dimensional numpy array of size n (dimension of the space) corresponding to the position of a point.
            It is the representation of the phenotype of the individual.
            Other parameters are self defined in the class object (see __init__) : the useful parameters here are
            alpha and Q.

        ------
        Return
            w : float
            fitness value of this particular phenotype. It is computed using an exponential fitness function from Tenaillon 2014. 
            With alpha = 0.5 and Q = 2, we get the classical Gaussian distribution. 

        """
        d = np.linalg.norm(position) # compute the euclidian distance to to the optimum
        w = np.exp(-self.alpha * d**self.Q) # Tenaillon 2014 (playing on the parameters alpha and Q modify fitness distribution in the phenotypic landscape)
        return w


    def extend_data(self, time : int) -> None:
        self.methods = np.concatenate((self.methods, np.full((time,len(self.mutation_functions)), fill_value= False)))
        self.positions = np.concatenate((self.positions, np.zeros(shape = (time,self.dimension))))
        self.mean_size = np.concatenate((self.mean_size, np.zeros(shape = time)))
        self.std_size = np.concatenate((self.std_size, np.zeros(shape = time)))
        self.fitness = np.concatenate((self.fitness, np.zeros(shape = time)))
        self.nb_genes = np.concatenate((self.nb_genes, np.zeros(shape = time)))
        self.modularities = np.concatenate((self.modularities, np.zeros(shape = time)))
        return

    def evolve_until_fitness(self, fitness_limit: float) -> None:
        time = self.current_time
        while self.current_fitness < fitness_limit:
            time += 1
            if time >= len(self.methods):
                self.extend_data(time)
            self.simulation_step(time)

            
        self.current_time = time
        return

    def evolve_successive(self, time_step : int) : # 20 sec
        """
        Main method that simulate the evolution for a certain time. 
        At each iteration, only make one kind of change in the genome (duplication, deletion, mutation), 
        modify the list of genes in consequence, compute the new phenotype and its fitness, 
        test if this modification in the genome is fixed and memorize some importants caracteristics 
        (fitness, fitness effect, position, ...) that will allow us to study the simulated evolution.

        Parameters
        ------
            time_step : int
                Number of time steps (successive mutations = 3*time_step) on which we want to iterate in the simulation

        ------
        Return :
            Actualize self paramaters, which are :
            memory : list
            Evolution of the position of the individual in the phenotypic space after each fixed modification of the genome
            fitness : list
            Evolution of the fitness of the actual phenotype at each iteration
            effects : list
            Fitness effects of the modification of the genome at each iteration
            methods : list
            If there as been a modification of the genome, memorize by which mechanism it has been changed (dupl and/or del and/or mut)
            nb_genes : list
            Evolution of the number of gene at each iteration
        
        """

        #Add extra space to the different vectors so they can fit new simulation data
        time_step = int(time_step)
        self.extend_data(time_step)

        time = self.current_time
        for t in range(time_step):
            time += 1

            if time == self.timestamp : # Save position, fitness and genotype for future comparisons
                self.timestamp_fitness = self.fitness.copy()
                self.timestamp_position = self.current_pos.copy()
                self.timestamp_genes = self.genes.copy()
            
            self.simulation_step(time)
        self.current_time = time

    def simulation_step(self, time : int) -> None:
        if self.display_fixation:
            print(f"Generation: {time}", end = "\r")

        mutated_genes = self.genes
        mutation_occured = np.full(len(self.mutation_functions), fill_value= False)
        for i, mutation in enumerate(self.mutation_functions):
            mutated_genes, mut = mutation(self.genes)
            if  mut and self.fixation_check(mutated_genes):
                # print(mutation)
                self.fixation(mutated_genes)
                mutation_occured[i] = True
                if mutation == self.duplication:
                    self.duplication_events[-1]['fixed'] = True
        
        if any(mutation_occured):
            self.methods[time] = np.array(mutation_occured)
            self.fitness[time] = self.current_fitness
            self.nb_genes[time] = len(self.genes)
            self.positions[time] = (self.current_pos)
            sizes = np.linalg.norm(self.genes, axis = 1)
            self.mean_size[time] = np.mean(sizes)
            self.std_size[time] = np.std(sizes)
            self.modularities[time] = self.find_modularity(self.genes)
        else:
            self.methods[time] = np.array(np.full(len(self.mutation_functions),False))
            self.fitness[time] = self.fitness[time - 1]
            self.nb_genes[time] = self.nb_genes[time - 1]
            self.positions[time] = self.positions[time - 1]
            self.mean_size[time] = self.mean_size[time - 1]
            self.std_size[time] = self.std_size[time- 1 ]
            self.modularities[time] = self.modularities[time - 1]

        return

    def fixation_check(self, new_genes: np.ndarray[np.ndarray[float]]):
        """
        Test if a change in the genotype will be fixed.
        
        Parameters
        ------
        
        new_genes : np.ndarray[list[float]]
            The list of genes of the genotype after the mutation/duplication/deletion 

        Return
        ------
        Fixation: bool
            True if fixation occurred, else False
            

        """
        new_pos = self.init_pos + np.sum(new_genes, axis = 0)

        
        # new_fitness  = self.fitness_calc(new_pos) # its new fitness
        if np.linalg.norm(new_pos) < np.linalg.norm(self.current_pos):
            return True
        else:
            return False

    def fixation(self,new_genes : np.ndarray[np.ndarray[float]]) -> None:
        """
        Fixates the parameter new_genes as the new genome
        
        Parameters
        -----
        new_genes : np.ndarray[np.ndarray[float]]
            The genes to be fixated
        
        Returns
        -----
        None

        """
        self.genes = new_genes
        self.current_pos = self.init_pos + np.sum(new_genes, axis = 0) 
        self.current_fitness = self.fitness_calc(self.current_pos)
        
        if self.display_fixation:
            print(f'\nNew genome fixated with fitness {self.current_fitness}, nr of genes: {len(self.genes)}')

        return

    def find_modularity(self, genes: np.ndarray) -> float:
        gene_copies = np.copy(genes)
        n = self.dimension  # Dimension of the vectors
        scale = 1 / np.sqrt(n)
        weighted_modularity_sum = 0
        total_weight = 0

        for gene in gene_copies:
            weight = np.linalg.norm(gene)
            gene /= weight  # Normalize the gene vector

            # Calculate the modularity of the gene
            strongest_dir = max(abs(gene))  # Max absolute value of the components
            modularity = (strongest_dir - scale) / (1 - scale)

            weighted_modularity_sum += modularity * weight
            total_weight += weight

        # Return the weighted average modularity
        M = weighted_modularity_sum / total_weight if total_weight != 0 else 0
        return M

    def get_graph(self) -> nx.Graph:
        edges = [(f'gene {i}',f'trait {j}',{'weight' : w[j]}) for (i,w) in enumerate(self.genes) for j in range(self.dimension)]
        graph = nx.from_edgelist(edges)

        return graph
    
    def get_communities(self) -> list[set]:
        graph = self.get_graph()
        partition = nx.community.louvain_communities(graph)

        return partition
    
    def reinitialize(self):
        self.genes = np.array([self.genes[-1]])
        self.init_pos = self.current_pos-self.genes[0]
        
        time = self.current_time
        self.nb_genes[time] = 1
        self.methods[time] = [False]*len(self.mutation_functions)
        self.fitness[time] = self.current_fitness
        self.nb_genes[time] = len(self.genes)
        self.positions[time] = (self.current_pos)
        sizes = np.linalg.norm(self.genes, axis = 1)
        self.mean_size[time] = np.mean(sizes)
        self.std_size[time] = np.std(sizes)
        self.modularities[time] = self.find_modularity(self.genes)


####################
#### Parameters ####
####################

if __name__ == "__main__":
    #Save a FisherGeometricObjectModel with parameters from Parameters.json to the file FisherObject.pkl


    with open("Parameters.json", 'rb') as input:
        fgm_args = json.load(input)
    fgm = FisherGeometricModel(**fgm_args)

    n_generations = 1*10**4
    fgm.evolve_successive(n_generations)
    with open('FisherObject', 'wb') as output:
        pickle.dump(fgm, output, pickle.HIGHEST_PROTOCOL)