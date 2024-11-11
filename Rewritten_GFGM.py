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


##############################
#### Functions definition ####
##############################

class FisherGeometricModel() :
    def __init__(self, n: int = 3, initial_distance: float = 20, mutation_methods : list[str] = ["multiplication","addition", "duplication", "deletion"], alpha: float = 0.5, Q: float = 2, sigma_point: float = 1, duplication_rate: float = 0.01, deletion_rate: float = 0.01, point_rate: float = 0.0001, ratio: float = 1, initial_gene_method: str = "random", timestamp : float = 1e5, sigma_mult : float = 0.1, addition_rate : float = 0.01, sigma_add : float = 0.1, multiplication_rate : float = 0.05, display_fixation : bool = False) :
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
            Ratio between sigma_gene and sigma_point (size of the first gene). Affects the rate of duplication versus mutation.
        sigma_point
            Standard deviation of the mutational vector on each axis
        duplication_rate
            Rate of duplication in nb/gene/generation
        deletion_rate
            Rate of deletion in nb/gene/generation 
        point_rate
            Rate of mutation in nb/genome/generation
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
        self.sigma_point : float = sigma_point
        self.sigma_mult : float = sigma_mult
        self.sigma_add : float = sigma_add

        self.init_pos : np.ndarray[float] = self.create_initial_starting_position(initial_distance)
        self.genes : np.ndarray[np.ndarray[float]]= np.array([self.create_fixed_first_gene(initial_gene_method)]) # chose the direction/size of the first gene 

        self.current_pos = self.init_pos + np.sum(self.genes, axis = 0) # the real phenotypic position of the individual is computed by adding the genes vectors to the initial position
        self.current_fitness = self.fitness_calc(self.current_pos) # Compute the fitness of the actual phenotype before any mutation happen

        self.multiplication_rate = multiplication_rate
        self.addition_rate = addition_rate
        self.duplication_rate = duplication_rate 
        self.deletion_rate = deletion_rate
        self.point_rate = point_rate

        self.mutation_functions = np.array([None]*len(mutation_methods))
        for i,method in enumerate(mutation_methods):
            if method == "multiplication":
                self.mutation_functions[i] = self.multiplicative_point_mutation
            elif method == "point":
                self.mutation_functions[i] = self.addative_point_mutation
            elif method == "addition":
                self.mutation_functions[i] = self.add_random_genes
            elif method == "duplication":
                self.mutation_functions[i] = self.duplication
            elif method == "deletion":
                self.mutation_functions[i] = self.deletion
            else:
                raise Warning(f"Unknown mutation method \"{method}\"")


        fit_1 = self.fitness_calc(self.init_pos)
        fit_2 = self.fitness_calc(self.current_pos)
        self.positions : np.ndarray[np.ndarray[float]] = [self.init_pos, self.current_pos] # memorize the to first position, which are the initial phenotype of the individual and its phenotype after addition of the first gene
        self.fitness : np.ndarray[float] = [fit_1, fit_2] # fitness values of the phenotype at each iteration
        self.methods = np.array([[False]*len(mutation_methods),[False]*len(mutation_methods)]) # method use to modificate the genotype at each iteration
        self.nb_genes = [0,1] # gene count at each generation
        self.mean_size = np.array([0,np.linalg.norm(self.genes[0])]) # mean size of the genes at each iteration
        self.std_size = np.array([0,0])

        #For saving a timestamp of the process
        self.timestamp = timestamp
        self.timestamp_fitness = np.array([]) # to memorize the list of fitness after 10**5 generations and resumes evolution at this step
        self.timestamp_position = np.zeros(n) # to memorize the position after 10**5 generations and resumes evolution at this step
        self.timestamp_genes = np.array([]) # to memorize the list of genes after 10**5 generations and resumes evolution at this step
        
        self.current_time = 1
        initial_signs = self.init_pos*self.genes[0]
        self.initial_beneficial_directions = np.sum([1 for i in range(n) if initial_signs[i] < 0])
        self.display_fixation = display_fixation

        self.args = {'n': n, 'initial_distance': initial_distance, 'mutation_methods': mutation_methods, 'alpha' : alpha, 'Q' : Q, 'sigma_point': sigma_point,  'duplication_rate': duplication_rate, 'deletion_rate': deletion_rate, 'point_rate': point_rate, 'initial_gene_method': initial_gene_method, 'timestamp': timestamp, 'sigma_mult': sigma_mult, 'addition_rate': addition_rate, 'multiplication_rate': multiplication_rate, 'sigma_add' : sigma_add, 'multiplication_rate' : multiplication_rate, 'display_fixation':display_fixation}
    
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
            Other parameters arre self defined in the class object (see __init__). Useful parameters are init_pos, sigma_point,
            dimension.

        ------
        Return : 
            gene : np.darray
            1-dimensional numpy array of size n (dimension of the phenotypic space) representing the gene.
 
        """

        if method == "random" :
            s = -1 
            while s < 0 : # we consider that the first gene must be at least neutral or beneficial so that the organism survive
                gene = np.random.normal(0, self.sigma_point, self.dimension) # We draw the gene as if it was a mutation of the starting point in the Standart FGM
                new_pos = self.init_pos + gene # compute the new phenotypic position after adding the gene 
                s = self.fitness_effect(self.fitness_calc(self.init_pos), self.fitness_calc(new_pos)) # compute the fitness effect of such a gene.
            # print(np.linalg.norm(gene))
            
        
        elif method == "parallel" : 
            # en partant sur une "diagonale" du graphe (cadrant positif)
            # return np.ones(self.dimension) * (-r * self.sigma_point  / np.sqrt(self.dimension)) # gene on the axe to the optimum (dans le cas où la position choisi est à 45°) : in this case the gene is optimal for adaptation, the simulation only make duplication 
            # attention dépend à nouveau bcp de la taille du premier gène. S'il est grand, ne fait que des duplications. Si il est petits, c'est assez aléatoire (il y a toujours pas mal de duplication au début mais pas tant que ça)
            # Dans ce cas comme les mutations sont assez bénéfiques aussi comparé à la duplication elles se fixent aussi ce qui bouge le point de l'axe optimal 
            
            grad = self.fitness_gradient(self.init_pos) # gradient of the fitness function

            s = -1
            while s < 0 : 
                gene = np.random.normal(0, 1, self.dimension)  # Random direction
                
                grad_norm = np.linalg.norm(grad)
                if grad_norm > 0:
                    grad_unit = grad / grad_norm # Normalize to unit length
                    gene = np.dot(gene, grad_unit) * grad_unit # Project the gene onto the axe of the gradient

                gene = gene / np.linalg.norm(gene)  # Normalize to unit length
                gene = gene * self.sigma_point  # Scale to the desired length

                new_pos = self.init_pos + gene
                s = self.fitness_effect(self.fitness_calc(self.init_pos), self.fitness_calc(new_pos)) # Compute the fitness effect (should be highly beneficial)
            
             
        
        elif method == "orthogonal" :
            # methode 1 : (en partant sur un axe du graphique)
            # gene = np.ones(self.dimension) * (-r * self.sigma_point / np.sqrt(self.dimension))
            # gene[0] *= 0 # 0 si on part sur un axe (mettre toutes les valeurs à 0 sauf 1 dans init_pos). Dans ce cas le gène est délétère : il est supprimer et rien ne peut se faire de mieux ensuite.
            # si on utilise cette méthode sans partir sur un axe, le gène est quasiment bien aligné, dans ce cas c'est un peu comme parallele : bcp de duplication au début, mais ensuite mute un peu et supprime les gènes mauvais ?
            
            # methode 2 : ( modifier deux directions suffit ?)
            # gene[0] -= r * self.sigma_point / sqrt(2)
            # gene[1] += r * self.sigma_point / sqrt(2)

            # methode 3 : calcul du gradient de f permettant de connaitre la direction de variation la plus forte de f = l'axe optimal
            grad = self.fitness_gradient(self.init_pos) # gradient of the fitness function
            gene = np.random.normal(0, 1, self.dimension)  # Random direction

            grad_norm = np.linalg.norm(grad)
            if grad_norm > 0:
                grad_unit = grad / grad_norm # Normalize to unit length
                gene = gene - np.dot(gene, grad_unit) * grad_unit # Project the gene onto the space orthogonal to the gradient

            gene = gene / np.linalg.norm(gene)  # Normalize to unit length
            gene = gene * self.sigma_point  # Scale to the desired length

            new_pos = self.init_pos + gene
            s = self.fitness_effect(self.fitness_calc(self.init_pos), self.fitness_calc(new_pos)) # compute its fitness effect (should be little deleterious)
            # print(s)
            # à voir ce qu'il se passerait si on autorisait pas la deletion du seul gène que l'individu possède.
            
             
        
        elif method == "only_one_deleterious_direction" :
            # Methode 1 : If the initial position is on a "diagonal" of the graph (positive quadrant)
            # gene = np.ones(self.dimension) * (-r * self.sigma_point / np.sqrt(self.dimension))
            # gene[0] *= -1

            # Methode 2 : with grad f :
            grad = self.fitness_gradient(self.init_pos) # gradient of the fitness function
            gene = np.random.normal(0, 1, self.dimension)  # Random direction

            grad_norm = np.linalg.norm(grad)
            if grad_norm > 0:
                grad_unit = grad / grad_norm # Normalize to unit length
                gene = np.dot(gene, grad_unit) * grad_unit # Project the gene onto the axe of the gradient

            gene = gene / np.linalg.norm(gene)  # Normalize to unit length
            gene = gene * self.sigma_point  # Scale to the desired length
            # Construct the gene as if it was parallel to the optimal axes
            
            gene[0] *= -1 # only make one dimension wrong

            new_pos = self.init_pos + gene
            s = self.fitness_effect(self.fitness_calc(self.init_pos), self.fitness_calc(new_pos)) # Compute the fitness effect (should be beneficial, but less than if it was optimal)
            # print(s)

             # mute a little to correct direction, then makes a lot of duplication, then delete some genes (+mute and duplicate) == get ride of the bad genes, then mute a lot
        
        elif method == "neutral" : # Move the initial position on the same isocline of fitness
            grad = self.fitness_gradient(self.init_pos) # gradient of the fitness function

            s = -1
            while s < 0 : # Because of rounding, the computer may not find a fitness effect of exactly 0. We then try to find at least a gene that is not deleterious
                random_vect = np.random.normal(0, 1, self.dimension)  # Random direction

                grad_norm = np.linalg.norm(grad)
                if grad_norm > 0:
                    grad_unit = grad / grad_norm # Normalize to unit length
                    random_vect = random_vect - np.dot(random_vect, grad_unit) * grad_unit # Project the vector onto the space orthogonal to the gradient
                    orthogonal_unit = random_vect / np.linalg.norm(random_vect) # unit vector of the axis orthogonal to the optimal axe
                
                z = self.sigma_point # Wanted size of the gene
                d = np.linalg.norm(self.init_pos) # Wanted distance to the optimum
                triangle_surface = z/2 * np.sqrt(d**2 - z**2 / 4) # Geometricaly, the gene vector and the axes linking the initial position / the position after adding the gene to the optimum form an isosceles triangle
                # there are different way of computing a triangle surface, which help us to find angles from the known distances
                sinus_gamma = 2*triangle_surface/d**2 
                gamma = np.arcsin(sinus_gamma) # angle between the optimal axes before and after adding the gene
                theta = (np.pi - gamma) / 2 # angle between the optimal axe and the gene vector

                gene = z*np.cos(theta)*grad_unit + z*np.sin(theta)*orthogonal_unit # coordinate of the gene vector using optimal axe and its orthogonal axe

                new_pos = self.init_pos + gene
                s = self.fitness_effect(self.fitness_calc(self.init_pos), self.fitness_calc(new_pos)) # Compute its fitness effect (should be 0 or very near)
            # print(s)

            
            # methode 2 : avec l'analyse mathématique sur les duplications benefiques : 
            # cos(theta) = z/(2*d) --> pas besoin d'utiliser le triangle isocèle
        
        elif method == "semi_neutral" : # the first gene is partially well adapted
            grad = self.fitness_gradient(self.init_pos) # gradient of the fitness function

            s = -1
            while s < 0 : # Because of rounding, the computer may not find a fitness effect of exactly 0. We then try to find at least a gene that is not deleterious
                random_vect = np.random.normal(0, 1, self.dimension)  # Random direction

                grad_norm = np.linalg.norm(grad)
                if grad_norm > 0:
                    grad_unit = grad / grad_norm # Normalize to unit length
                    random_vect = random_vect - np.dot(random_vect, grad_unit) * grad_unit # Project the vector onto the space orthogonal to the gradient
                    orthogonal_unit = random_vect / np.linalg.norm(random_vect) # unit vector of the axis orthogonal to the optimal axe
                
                z = self.sigma_point # Wanted size of the gene
                d = np.linalg.norm(self.init_pos) # Wanted distance to the optimum
                #QUESTION: How does this triangle equation work? Vsiualisation?
                triangle_surface = z/2 * np.sqrt(d**2 - z**2 / 4) # Geometricaly, the gene vector and the axes linking the initial position / the position after adding the gene to the optimum form an isosceles triangle
                # there are different way of computing a triangle surface, which help us to find angles from the known distances
                sinus_gamma = 2*triangle_surface/d**2 
                gamma = np.arcsin(sinus_gamma) # angle between the optimal axes before and after adding the gene
                theta = (np.pi - gamma) / 2 # angle between the optimal axe and the gene vector if the gene is neutral
                theta /= 2 # semi neutral gene : its angle is divided by 2 by comparison to a neutral gene

                gene = z*np.cos(theta)*grad_unit + z*np.sin(theta)*orthogonal_unit # coordinate of the gene vector using optimal axe and its orthogonal axe

                new_pos = self.init_pos + gene
                s = self.fitness_effect(self.fitness_calc(self.init_pos), self.fitness_calc(new_pos)) # Compute its fitness effect (should be positive but not very large (less than for parallel and only_one_deleterious))
            # print(s)

        return gene

    def add_random_genes(self,base_genes : np.ndarray[np.ndarray[float]]):
        """
        Adds random genes drawn form a normal distribution to a previous list of genes.

        Parameters
        -----
        base_genes : np.ndarray[np.ndarray[float]]
            List of the genes to be modified
        sigma : float
            Variance of the added gene
        rate : float
            Rate of addition. This determines the average number of genes added.

        Returns
        -----
        new_genes : np.ndarray[np.ndarray[float]]
            List of genes including the newly added one
        """
        list_genes = base_genes.copy()
        nb_additions = np.random.poisson(self.addition_rate)
        if nb_additions > 0:
            genes = np.random.normal(0, self.sigma_add, (nb_additions,self.dimension))
            list_genes = np.concatenate([list_genes, genes])
        return list_genes, nb_additions > 0

    def fitness_gradient(self, position : np.array) -> np.array:
        """
        Compute the gradient vector of the fitness function at a given position, 
        which as the same direction as the optimal axe linking the position to the optimum. 
        (Direction of maximal variation of fitness in the neighborhood of the point)

        ------
        Parameters :
            position
            One dimensional numpy array of size self.dimension representing the actual position in the phenotypic space.
            Other parameters are self defined in the class object (see __init__). Useful parameters here are
            alpha and Q. 

        ------
        Return : 
            np.array
            One dimensional Numpy array of size self.dimension representing the gradient vector 
            of the fitness function, which is just the vector of the partial derivative of the function.
        """
        # return -self.Q * self.alpha * position * self.fitness_calc(position) # only true when Q = 2 
        return -self.Q * self.alpha * position * np.linalg.norm(position)**(self.Q - 2) * self.fitness_calc(position)

    def multiplicative_point_mutation(self, current_genes : np.ndarray[np.ndarray[float]]):
        """
        Randomly mutate every genes in the genotype by multiplying a gaussian noise to them. The mutation
        may differ from one gene to another. We consider in this case that the mutation rate is per genome
        and have a value of 1, meaning that each gene in the genome gets mutated exactly once per generation.

        Parameters
        ------
        current_genes : np.ndarray[np.ndarray[float]]
            List of genes to be duplicated
        The useful parameters here are dimension, sigma_point

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
        m = np.random.lognormal(0,self.multiplication_rate, size = (n, self.dimension))
        # m = np.random.exponential(1, size = (n, self.dimension))
        new_genes = [new_genes[i] * m[i] for i in range(n)] # modify every genes in the list by adding the corresponding mutation. All genes do not mutate the same way
        # print(f"1st gene: {self.genes[0]}, mutation: {m[0]}")
        mut = True
        return new_genes, mut
    
    def addative_point_mutation(self, current_genes : np.ndarray[np.ndarray[float]]):
        """
        Randomly mutate every genes in the genotype by adding a gaussian noise to them. The mutation
        may differ from one gene to another. We consider in this case that the mutation rate is per genome
        and have a value of 1, meaning that each gene in the genome gets mutated exactly once per generation.

        Parameters
        ------
        current_genes : np.ndarray[np.ndarray[float]]
            List of genes to be duplicated
        The useful parameters here are dimension, sigma_point

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
        m = np.random.normal(0, self.sigma_point, size=(n, self.dimension)) # draw the mutation from a normal distribution of variance n*sigma_point**2 (variance of a sum of mutation)
        
        new_genes = [new_genes[i] + m[i] for i in range(n)] # modify every genes in the list by adding the corresponding mutation. All genes do not mutate the same way
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
        nb_dupl = np.random.poisson(self.duplication_rate*n) # number of duplication to do. the rate is multiply by the number of gene and the size of the population so that it represent the number of duplication per generation.

        if nb_dupl > 0:
            actual_duplications = min(n, nb_dupl)
            added_gene_index = np.random.choice(range(n),actual_duplications,replace = True)
            added_genes = list_genes[added_gene_index]
            list_genes = np.concatenate((list_genes,added_genes))
            
            # print(f"nr dupl:{nb_dupl}, indices: {added_gene_index}, added genes : {added_genes}, total list : {list_genes}")
        return list_genes, nb_dupl > 0
    
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
        nb_del = np.random.poisson(self.deletion_rate*n) # number of duplication to do. the rate is multiply by the number of gene and the size of the population so that it represent the number of duplication per generation.

        if nb_del > 0:
            actual_deletions = min(n, nb_del)
            removed_genes_index = np.random.choice(range(n),actual_deletions,replace = False)
            list_genes = np.delete(list_genes,removed_genes_index,0)

        return list_genes, nb_del > 0
    
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
    
    def fitness_effect(self, previous_fitness, new_fitness):
        """
        Compute the fitness effect of a change in the phenotype/genotype.
        It is just the logarithm of the ratio between the fitness of the new position in the space 
        and the fitness of the position before the modification. (ancestral position).

        ------
        Parameters :
            previous_fitness : float
            fitness of the ancestral phenotype/genotype.
            new_fitness : float
            fitness of the new phenotype/genotype.

        ------
        Return
            np.log(new_fitness/previous_fitness) : float
            Fitness effect s of a modification in the phenotype/genotype of an individual (due to 
            mutation, duplication, deletion)

        """

        s = np.log(new_fitness/previous_fitness)
        return s
        # when the new fitness is higher than the ancestral one, the ratio is > 1, so the effect is > 0
        # return new_fitness/previous_fitness - 1 # Martin 2006 (for s small, not used here)
        # if > 0, the mutation as improve the fitness, it is beneficial
    
    def fixation_probability(self, s: float) -> float:
        """
        Compute the fixation probability of a given modification in the genome, 
        having a fitness effect s.
        Here we consider a population of infinite size, which means that drift is negligeable 
        in front of selection and that only beneficial mutation will have the possibility to fix. 

        ------
        Parameters
            s,
                fitness effect of the modification (see function fitness_effect())

        ------
        Return
            p,
                Probability of fixation of a mutation/duplication/deletion. 
            
        """
        if s > 0 : # beneficial mutation
            p = (1 - np.exp(-2*s)) # Barrett 2006 
        else : # deleterious mutation
            p = 0 # deleterious mutation do not fix
        return p

    def evolve_until_fitness(self, fitness_limit: float) -> None:
        time = self.current_time
        while self.current_fitness < fitness_limit:
            if time >= len(self.methods):
                self.methods = np.concatenate((self.methods, np.full((time,4), fill_value= False)))
                self.positions = np.concatenate((self.positions, np.zeros(shape = (time,self.dimension))))
                self.mean_size = np.concatenate((self.mean_size, np.zeros(shape = time)))
                self.std_size = np.concatenate((self.std_size, np.zeros(shape = time)))
                self.fitness = np.concatenate((self.fitness, np.zeros(shape = time)))
                self.nb_genes = np.concatenate((self.nb_genes, np.zeros(shape = time)))
            self.simulation_step(time)

            time += 1
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
        self.methods = np.concatenate((self.methods, np.full((time_step,4), fill_value= False)))
        self.positions = np.concatenate((self.positions, np.zeros(shape = (time_step,self.dimension))))
        self.mean_size = np.concatenate((self.mean_size, np.zeros(shape = time_step)))
        self.std_size = np.concatenate((self.std_size, np.zeros(shape = time_step)))
        self.fitness = np.concatenate((self.fitness, np.zeros(shape = time_step)))
        self.nb_genes = np.concatenate((self.nb_genes, np.zeros(shape = time_step)))

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
            mutated_genes, mutation_occured[i] = mutation(mutated_genes)
        
        if self.fixation_check(mutated_genes):
            self.fixation(mutated_genes)
            self.methods[time] = np.array(mutation_occured)
        else:
            self.methods[time] = np.array([False,False,False,False])

        self.fitness[time] = self.current_fitness
        self.nb_genes[time] = len(self.genes)
        self.positions[time] = (self.current_pos)
        sizes = np.linalg.norm(self.genes, axis = 1)
        self.mean_size[time] = np.mean(sizes)
        self.std_size[time] = np.std(sizes)

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

        new_fitness  = self.fitness_calc(new_pos) # its new fitness
        s = self.fitness_effect(self.current_fitness, new_fitness) # the fitness effect of the modification
        prob = self.fixation_probability(s) # its fixation probality depending on its fitness effect

        if np.random.rand() < prob : # the mutation is fixed
            # print(f"\nfixation succeded, probability {prob}")
            return True
            
        else : # if the mutation didn't fix
            # print(f"\nfixation failed ,  probability {prob}")
            
            return False

    def fixation(self,new_genes : np.ndarray[np.ndarray[float]]) -> None:
        """
        Fixates the parameter new_genes as the new genome
        
        Parameters
        -----
        new_genes : np.ndarray[list[float]]
            The genes to be fixated
        display : bool, optional
            If True, prints that a new genome has been fixed. Default is True
        
        Returns
        -----
        None
            If display == True will print that a new genome has been fixated and its fitness.

        """
        self.genes = new_genes
        self.current_pos = self.init_pos + np.sum(new_genes, axis = 0) 
        self.current_fitness = self.fitness_calc(self.current_pos)
        
        if self.display_fixation:
            print(f'\nNew genome fixated with fitness {self.current_fitness}, nr of genes: {len(self.genes)}')

        return

    def get_graph(self) -> nx.Graph:
        edges = [(f'gene {i}',f'trait {j}',{'weight' : w[j]}) for (i,w) in enumerate(self.genes) for j in range(self.dimension)]
        graph = nx.from_edgelist(edges)

        return graph
    
    def get_communities(self) -> list[set]:
        graph = self.get_graph()
        partition = nx.community.louvain_communities(graph)

        return partition


####################
#### Parameters ####
####################

if __name__ == "__main__":
    #Save a FisherGeometricObjectModel to the file FisherObject.pkl

    dimension = 3  # Number of traits in the phenotype space n
    initial_distance = 10 # initial distance to the optimum
    # sigma_point = r/np.sqrt(n_traits) # Standard deviation of the mutation effect size # Tenaillon 2014
    sigma_point = 0.01 # énormement de duplication/deletion par rapport au nombre de mutation quand on baisse sigma (voir sigma=0.01)
    sigma_mult = 0.06
    alpha = 1/2
    Q = 2

    multiplication_rate = 5e-2
    addition_rate = 1e-2
    point_rate = 1e-4 # rate of mutation mu
    duplication_rate = 1e-2 # /gene/generation
    deletion_rate = 1e-2 # /gene/generation

    initial_gene_method = "parallel"
    mutation_methods : list[str] = ["multiplication","addition", "duplication", "deletion"]

    display_fixation = True
    fgm_args = {'display_fixation':display_fixation, 'n': dimension, 'initial_distance': initial_distance, 'mutation_methods': mutation_methods, 'sigma_point': sigma_point,  'duplication_rate': duplication_rate, 'deletion_rate': deletion_rate, 'point_rate': point_rate, 'initial_gene_method': initial_gene_method, 'sigma_mult': sigma_mult, 'addition_rate': addition_rate, 'multiplication_rate': multiplication_rate}
    fgm = FisherGeometricModel(**fgm_args)

    n_generations = 5*10**4
    fgm.evolve_successive(n_generations)
    with open('FisherObject', 'wb') as output:
        pickle.dump(fgm, output, pickle.HIGHEST_PROTOCOL)