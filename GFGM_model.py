'''
Genotypic Fisher Geometric Model of Adaptation.
'''


import numpy as np
import pickle
import json

class FisherGeometricModel() :
    def __init__(self, 
                 n: int = 3, initial_distance: float = 20, 
                 mutation_methods : list[str] = ["addition", "duplication", "deletion"], 
                 sigma_add: float = 1, duplication_rate: float = 0.01, 
                 deletion_rate: float = 0.01, initial_gene_method: str = "random", 
                 display_fixation : bool = False,
                 save_unfixed_mutations : bool = False):
        """
        Initialized the main parameters and variables used in our modified FGM.
        
        Parameters
        ------
        n 
            Dimension of the phenotypic space = number of main indepedent phenotypic traits that are used as axis is the model
        intial_position
            Initial phenotype/genotype = position in the space of the individual.
        sigma_add
            Standard deviation of the mutational vector on each axis
        duplication_rate
            Rate of duplication in nb/gene/generation
        deletion_rate
            Rate of deletion in nb/gene/generation 
        initial_gene_method
            Method of choosing the initial gene
        mutation_methods
            List of mutation methods to be used
        display_fixation
            If True, display the fixation of the genome
        save_unfixed_mutations
            If True, save the unfixed mutations in the mutation_events dictionary (memory intensive for large simulations)
        """
        self.dimension : int = n
        self.optimum : np.ndarray[float] = np.zeros(n) #define the origin as the fitness optimum
        self.sigma_add : float = sigma_add

        self.init_pos : np.ndarray[float] = self.create_initial_starting_position(initial_distance)
        self.genes : np.ndarray[np.ndarray[float]]= np.array([self.create_fixed_first_gene(initial_gene_method)]) # chose the direction/size of the first gene 

        self.current_pos = self.init_pos + np.sum(self.genes, axis = 0) # the real phenotypic position of the individual is computed by adding the genes vectors to the initial position
        self.current_distance = np.linalg.norm(self.current_pos) # Compute the distance to the optimum of the actual phenotype before any mutation happen

        self.duplication_rate : float = duplication_rate 
        self.deletion_rate : float = deletion_rate

        self.mutation_functions = np.array([None]*len(mutation_methods))
        for i,method in enumerate(mutation_methods):
            if method == "addition":
                self.mutation_functions[i] = self.addative_point_mutation
            elif method == "duplication":
                self.mutation_functions[i] = self.duplication
            elif method == "deletion":
                self.mutation_functions[i] = self.deletion
            else:
                raise Warning(f"Unknown mutation method \"{method}\"")


        self.positions : np.ndarray[np.ndarray[float]] = [self.init_pos, self.current_pos]
        self.mutation_events : dict[list[dict]]= {method : [] for method in mutation_methods}
        self.mutation_methods = mutation_methods
        self.total_genome_range = [0,np.linalg.norm(self.genes[0])] # total genome range is the sum of the norm of all the genes in the genome
        self.current_time = 1

        initial_signs = self.init_pos*self.genes[0]
        self.initial_beneficial_directions = np.sum([1 for direction in initial_signs if direction < 0])

        self.display_fixation = display_fixation
        self.save_unfixed_mutations = save_unfixed_mutations
        self.nb_genes = None # See get_nb_genes method

        self.args = {'n': n, 'initial_distance': initial_distance, 'mutation_methods': mutation_methods, 'duplication_rate': duplication_rate, 'deletion_rate': deletion_rate, 'initial_gene_method': initial_gene_method, 'sigma_add' : sigma_add, 'display_fixation':display_fixation, 'save_unfixed_mutations': save_unfixed_mutations, 'initial_beneficial_directions': self.initial_beneficial_directions}
    
    def get_args(self):
        return self.args

    def create_initial_starting_position(self, distance : float) -> np.ndarray[float]:
        initial_position = np.zeros(self.dimension)
        initial_position[0] = distance
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
            gene = -self.init_pos / np.linalg.norm(self.init_pos) * abs(np.random.normal(0,self.sigma_add)) # gene on the axe to the optimum             
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
        
        Return
        ------ 
        list_genes : np.ndarray[np.ndarray[float]]
            List of 1-dimensional numpy array of size n (dimension of the phenotypic space) representing the genes
            after mutation.
        mut : boolean
            Always put at true in this version because all genes are always mutated in each generation
        """
        new_genes = current_genes.copy()
        n = len(new_genes)
        m = np.random.normal(0, self.sigma_add, size=(n, self.dimension))
        
        new_genes = np.array([new_genes[i] + m[i] for i in range(n)])
        self.mutation_events['addition'].append({'time': self.current_time, 'mutation': m, 'fixed': False})
        
        return new_genes, True
   
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
            self.mutation_events['duplication'].append({'time': self.current_time, 'gene': added_gene, 'index' : added_gene_index, 'pos': self.current_pos, 'fixed': False})    
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
            self.mutation_events["deletion"].append({'time': self.current_time, 'gene': list_genes[removed_gene], 'index': removed_gene, 'pos': self.current_pos, 'fixed': False})
            list_genes = np.delete(list_genes,removed_gene,0)

        return list_genes, deletion_occurred

    def get_nb_genes(self):
        if self.nb_genes != None and len(self.nb_genes) == self.current_time:
            return self.nb_genes
        elif self.mutation_methods == ["addition"]:
            self.nb_genes = np.ones(self.current_time)
            return self.nb_genes            
        else: 
            events = np.zeros(self.current_time)
            events[0] = 1
            
            duplication_index = [event["time"] for event in self.mutation_events["duplication"] if event['fixed']]
            events[duplication_index] = 1
            
            if "deletion" in self.mutation_events:
                deletion_index = [event["time"] for event in self.mutation_events["deletion"] if event['fixed']]    
                events[deletion_index] -= 1

            self.nb_genes = np.cumsum(events)

            return self.nb_genes
    
    def extend_data(self, time : int) -> None:
        self.positions = np.concatenate((self.positions, np.zeros(shape = (time,self.dimension))))
        self.total_genome_range = np.concatenate((self.total_genome_range, np.zeros(shape = time)))
        return

    def evolve_until_distance(self, distance_limit: float) -> None:
        if distance_limit <= 0:
            return
        while self.current_distance > distance_limit:
            if self.current_time >= len(self.positions)-1:
                self.extend_data(self.current_time)
            self.simulation_step()

            
        return

    def evolve_successive(self, time_step : int):
        """
        Main method that simulate the evolution for a certain time. 
        At each iteration, only make one kind of change in the genome (duplication, deletion, mutation), 
        modify the list of genes in consequence, compute the new phenotype and its distance to the optimum, 
        test if this modification in the genome is fixed and memorize some importants caracteristics 
        (distance, distance effect, position, ...) that will allow us to study the simulated evolution.

        Parameters
        ------
            time_step : int
                Number of time steps (successive mutations = 3*time_step) on which we want to iterate in the simulation

        ------
        Return :
            Actualize self paramaters, which are :
            memory : list
            Evolution of the position of the individual in the phenotypic space after each fixed modification of the genome
            distance : list
            Evolution of the distance to the optimum of the actual phenotype at each iteration
            effects : list
            distance effects of the modification of the genome at each iteration
            methods : list
            If there as been a modification of the genome, memorize by which mechanism it has been changed (dupl and/or del and/or mut)
        
        """

        time_step = int(time_step)
        self.extend_data(time_step)

        for t in range(time_step):            
            self.simulation_step()

    def simulation_step(self) -> None:
        self.current_time += 1
        time = self.current_time
        if self.display_fixation:
            print(f"Generation: {time}", end = "\r")

        mutated_genes = self.genes
        any_mutation_fixed = False
        for i, mutation in enumerate(self.mutation_functions):
            mutated_genes, mutation_happened = mutation(self.genes)
            if  mutation_happened and self.fixation_check(mutated_genes):
                # print(mutation)
                self.fixation(mutated_genes)
                any_mutation_fixed = True
                self.mutation_events[self.mutation_methods[i]][-1]['fixed'] = True
            elif mutation_happened and not self.save_unfixed_mutations:
                self.mutation_events[self.mutation_methods[i]].pop(-1)

        
        if any_mutation_fixed:
            self.total_genome_range[time] = np.sum(np.linalg.norm(self.genes, axis = 1))
            self.positions[time] = self.current_pos
        else:
            self.total_genome_range[time] = self.total_genome_range[time - 1]
            self.positions[time] = self.positions[time - 1]
        
        
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

        
        if np.linalg.norm(new_pos) < self.current_distance:
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
        self.current_distance = np.linalg.norm(self.current_pos)
        
        if self.display_fixation:
            print(f'\nNew genome fixated with distance {self.current_distance}, nr of genes: {len(self.genes)}')

        return
    
    def reinitialize(self):
        self.genes = np.array([self.genes[-1]])
        self.init_pos = self.current_pos-self.genes[0]
        
        time = self.current_time
        self.positions[time] = (self.current_pos)

if __name__ == "__main__":
    #Save a FisherGeometricObjectModel with parameters from Parameters.json to the file FisherObject.pkl

    with open("Parameters.json", 'rb') as input:
        fgm_args = json.load(input)
    fgm = FisherGeometricModel(**fgm_args)

    ##uncomment top or bottom two to run the simulation
    # distance_limit = 0.1
    # fgm.evolve_until_distance(distance_limit)
    n_generations = 1*10**4
    fgm.evolve_successive(n_generations)
    with open('test_data/FisherObject', 'wb') as output:
        pickle.dump(fgm, output, pickle.HIGHEST_PROTOCOL)