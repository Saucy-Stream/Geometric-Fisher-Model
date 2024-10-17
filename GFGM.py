'''
Genotypic Fisher Geometric Model of Adaptation, 
Iterative version.
'''

##########################  
#### Import Libraries ####
##########################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from multiprocessing import Pool

##############################
#### Functions definition ####
##############################

class FisherGeometricModel() :
    def __init__(self, n: int, initial_position: list[float], alpha: float, Q: float, sigma_mut: float, duplication_rate: float, deletion_rate: float, point_rate: float, ratio: float, method: str, timestamp = 1e5) :
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
            Ratio between sigma_gene and sigma_mut (size of the first gene). Affects the rate of duplication versus mutation.
        sigma_mut
            Standard deviation of the mutational vector on each axis
        duplication_rate
            Rate of duplication in nb/gene/generation
        deletion_rate
            Rate of deletion in nb/gene/generation 
        point_rate
            Rate of mutation in nb/genome/generation
        method
            Method of choosing the initial gene
        timestamp
            Time to save the process
        
        Return
        ------ 
            None
            (memorize paramters and variables for later use in other methods)  
        """
        self.dimension : int = n
        self.optimum : list[float] = np.zeros(n) #define the origin as the fitness optimum
        self.alpha : float = alpha
        self.Q : float = Q 
        
        self.sigma_mut : float = sigma_mut

        self.init_pos : list[float] = initial_position

        self.genes : np.ndarray[np.ndarray[float]]= np.array([self.create_fixed_first_gene(ratio, method)]) # chose the direction/size of the first gene 

        self.current_pos = self.init_pos + np.sum(self.genes, axis = 0) # the real phenotypic position of the individual is computed by adding the genes vectors to the initial position
        self.current_fitness = self.fitness_calc(self.current_pos) # Compute the fitness of the actual phenotype before any mutation happen

        self.duplication_rate = duplication_rate 
        self.deletion_rate = deletion_rate
        self.point_rate = point_rate

        fit_1 = self.fitness_calc(self.init_pos)
        fit_2 = self.fitness_calc(self.current_pos)
        self.memory = [self.init_pos, self.current_pos] # memorize the to first position, which are the initial phenotype of the individual and its phenotype after addition of the first gene
        self.fitness = [fit_1, fit_2] # fitness values of the phenotype at each iteration
        self.methods = [] # method use to modificate the genotype at each iteration
        self.nb_genes = [0,1] # gene count at each generation
        self.mean_size = [0,np.linalg.norm(self.genes[0])] # mean size of the genes at each iteration
        self.std_size = [0,0]

        #For saving a timestamp of the process
        self.timestamp = timestamp
        self.memory_fitness = [] # to memorize the list of fitness after 10**5 generations and resumes evolution at this step
        self.memory_position = np.zeros(n) # to memorize the position after 10**5 generations and resumes evolution at this step
        self.memory_genes = [] # to memorize the list of genes after 10**5 generations and resumes evolution at this step
        
        self.figures = [] #For storing all avalible plots
    

    def create_fixed_first_gene(self, r: float, method: str):
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
            Other parameters arre self defined in the class object (see __init__). Useful parameters are init_pos, sigma_mut,
            dimension.

        ------
        Return : 
            gene : np.darray
            1-dimensional numpy array of size n (dimension of the phenotypic space) representing the gene.
 
        """

        if method == "random" :
            s = -1 
            while s < 0 : # we consider that the first gene must be at least neutral or beneficial so that the organism survive
                gene = np.random.normal(0, r*self.sigma_mut, self.dimension) # We draw the gene as if it was a mutation of the starting point in the Standart FGM
                new_pos = self.init_pos + gene # compute the new phenotypic position after adding the gene 
                s = self.fitness_effect(self.fitness_calc(self.init_pos), self.fitness_calc(new_pos)) # compute the fitness effect of such a gene.
            # print(np.linalg.norm(gene))
            
        
        elif method == "parallel" : 
            # en partant sur une "diagonale" du graphe (cadrant positif)
            # return np.ones(self.dimension) * (-r * self.sigma_mut  / np.sqrt(self.dimension)) # gene on the axe to the optimum (dans le cas où la position choisi est à 45°) : in this case the gene is optimal for adaptation, the simulation only make duplication 
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
                gene = gene * r * self.sigma_mut  # Scale to the desired length

                new_pos = self.init_pos + gene
                s = self.fitness_effect(self.fitness_calc(self.init_pos), self.fitness_calc(new_pos)) # Compute the fitness effect (should be highly beneficial)
            print(s)
            
             
        
        elif method == "orthogonal" :
            # methode 1 : (en partant sur un axe du graphique)
            # gene = np.ones(self.dimension) * (-r * self.sigma_mut / np.sqrt(self.dimension))
            # gene[0] *= 0 # 0 si on part sur un axe (mettre toutes les valeurs à 0 sauf 1 dans init_pos). Dans ce cas le gène est délétère : il est supprimer et rien ne peut se faire de mieux ensuite.
            # si on utilise cette méthode sans partir sur un axe, le gène est quasiment bien aligné, dans ce cas c'est un peu comme parallele : bcp de duplication au début, mais ensuite mute un peu et supprime les gènes mauvais ?
            
            # methode 2 : ( modifier deux directions suffit ?)
            # gene[0] -= r * self.sigma_mut / sqrt(2)
            # gene[1] += r * self.sigma_mut / sqrt(2)

            # methode 3 : calcul du gradient de f permettant de connaitre la direction de variation la plus forte de f = l'axe optimal
            grad = self.fitness_gradient(self.init_pos) # gradient of the fitness function
            gene = np.random.normal(0, 1, self.dimension)  # Random direction

            grad_norm = np.linalg.norm(grad)
            if grad_norm > 0:
                grad_unit = grad / grad_norm # Normalize to unit length
                gene = gene - np.dot(gene, grad_unit) * grad_unit # Project the gene onto the space orthogonal to the gradient

            gene = gene / np.linalg.norm(gene)  # Normalize to unit length
            gene = gene * r * self.sigma_mut  # Scale to the desired length

            new_pos = self.init_pos + gene
            s = self.fitness_effect(self.fitness_calc(self.init_pos), self.fitness_calc(new_pos)) # compute its fitness effect (should be little deleterious)
            # print(s)
            # à voir ce qu'il se passerait si on autorisait pas la deletion du seul gène que l'individu possède.
            
             
        
        elif method == "only_one_deleterious_direction" :
            # Methode 1 : If the initial position is on a "diagonal" of the graph (positive quadrant)
            # gene = np.ones(self.dimension) * (-r * self.sigma_mut / np.sqrt(self.dimension))
            # gene[0] *= -1

            # Methode 2 : with grad f :
            grad = self.fitness_gradient(self.init_pos) # gradient of the fitness function
            gene = np.random.normal(0, 1, self.dimension)  # Random direction

            grad_norm = np.linalg.norm(grad)
            if grad_norm > 0:
                grad_unit = grad / grad_norm # Normalize to unit length
                gene = np.dot(gene, grad_unit) * grad_unit # Project the gene onto the axe of the gradient

            gene = gene / np.linalg.norm(gene)  # Normalize to unit length
            gene = gene * r * self.sigma_mut  # Scale to the desired length
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
                
                z = r * self.sigma_mut # Wanted size of the gene
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
                
                z = r * self.sigma_mut # Wanted size of the gene
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

    def mutation_on_one_gene(self, list_genes : list) -> list[list, bool]:

        """
        Randomly mutate some genes in the genotype by adding a gaussian noise to them. The number of
        genes to mutate is drawn from a poisson distribution of parameter the mutation rate.

        ------
        Parameters :
            list_genes, 
            list of 1-D numpy arrays representing genes. It may differ from self.genes becasue some might have been duplicated or deleted. 
            Other parameters are self defined in the class object (see __init__). The useful parameters here are 
            dimension, sigma_mut, point_rate.

        ------
        Return : 
            list_genes, 
            list of 1-dimensional numpy array of size n (dimension of the phenotypic space) representing the genes
            after mutation.\\
            nb_mut > 0, True if there is at least one mutation

        """
        n = len(list_genes)
        nb_mut = np.random.poisson(self.point_rate * n) # Blanquart 2014 ; number of mutation to do. the rate is multiplied by the size of the population so that it represent the number of mutation per generation.
        # the mutation rate is a rate per gene, so we also need to multiply it by the number of gene
        list_genes = np.array(list_genes) # convert the list of genes into a numpy array to allow operation on arrays.
        if nb_mut > 0:
            #FIXME: This method of getting indices allows for repeats, so the number of mutations will likely be less than nb_mut
            indices = np.random.randint(0, n, nb_mut) # randomly choose the genes to mutate
            mutations = np.random.normal(0, self.sigma_mut, (nb_mut, self.dimension)) # Tenaillon 2014, Blanquart 2014 ; draw the mutation from a normal distribution of variance sigma_mut**2
            # devrait on faire nb_mut*self.sigma_mut (voir mutation_on_every_gene)? NON ?
            
            list_genes[indices] = list_genes[indices] + mutations # modify the corresponding genes by adding the mutation vector to them.

        return list_genes.tolist(), nb_mut > 0 # convert the list of genes back to a list (necessary because some operations use in the class only work on lists)

    def mutation_on_every_gene(self, current_genes : np.ndarray[np.ndarray[float]]):
        """
        Randomly mutate every genes in the genotype by adding a gaussian noise to them. The mutation
        may differ from one gene to another. We consider in this case that the mutation rate is per genome
        and have a value of 1, meaning that each gene in the genome gets mutated exactly once per generation.

        Parameters
        ------
        current_genes : np.ndarray[np.ndarray[float]]
            List of genes to be duplicated
        The useful parameters here are dimension, sigma_mut

        ------
        Return : 
            list_genes : list
            List of 1-dimensional numpy array of size n (dimension of the phenotypic space) representing the genes
            after mutation.
            mut : boolean
            Always put at true in this version because there is 1 mutation per generation (iteration)

        """
        new_genes = current_genes.copy()
        n = len(new_genes)
        m = np.random.normal(0, self.sigma_mut, size=(n, self.dimension)) # draw the mutation from a normal distribution of variance n*sigma_mut**2 (variance of a sum of mutation)
        
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

        ------
        Return
            list_genes : list
            list of 1-D numpy arrays (genes) modified after deletion of some genes.
            nb_del > 0 : boolean
            True if a deletion have been made.

        """
        n = len(current_genes)
        list_genes = np.array(current_genes.copy()) # make a copy of the initial list of genes so that it is not changed if the modification are not fixed afterward
        nb_del = np.random.poisson(self.deletion_rate*n) # number of duplication to do. the rate is multiply by the number of gene and the size of the population so that it represent the number of duplication per generation.

        if nb_del > 0:
            actual_deletions = min(len(list_genes), nb_del)
            removed_genes_index = np.random.choice(range(n),actual_deletions,replace = False)
            list_genes = np.delete(list_genes,removed_genes_index)

        return list_genes, nb_del > 0
    
    def fitness_calc(self, position): # 4sec
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

    def evolve_successive(self, time_step : int, case : str) : # 20 sec
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
            case : str
                The mutational method we want to use (mutation affect one gene (one_gene), the whole genome (all_gene), the whole genome at every iteration (always_all_gene))
        The useful class parameters here are init_pos, current_pos, genes

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
        for t in range(time_step):
            if t == self.timestamp : # Save position, fitness and genotype for future comparisons
                self.memory_fitness = self.fitness.copy()
                self.memory_position = self.current_pos.copy()
                self.memory_genes = self.genes.copy()

            print(f"Generation: {t}", end = "\r")

            duplicated_genes, duplication_occured = self.duplication(self.genes) # test if there are some duplications to do

            deleted_genes, deletion_occurred = self.deletion(duplicated_genes) # test if there are some deletions to do
            
            if case == "one_gene":
                point_genes, point_ocurred = self.mutation_on_one_gene(deleted_genes)
            else :
                point_genes, point_ocurred = self.mutation_on_every_gene(deleted_genes) 
            
            if self.fixation_check(point_genes):
                self.fixation(point_genes)
                self.methods.append(np.array([duplication_occured,deletion_occurred,point_ocurred]))
            else:
                self.methods.append(np.array([False,False,False]))

            self.fitness.append(self.current_fitness)
            self.nb_genes.append(len(self.genes))
            self.memory.append(self.current_pos)
            sizes = [np.linalg.norm(gene) for gene in self.genes]
            self.mean_size.append(np.mean(sizes))
            self.std_size.append(np.std(sizes))
    
    def pick_mutation_type(self) -> str:
        """
        Randomly selects which mutation event that will occurr during a generation

        Parameters
        -----
        None
        
        Returns
        -----
        genome_event
            str, one of 'duplication', 'deletion', 'point'.
        """

        events = np.array(['duplication', 'deletion', 'point'])
        rates = np.array([self.duplication_rate, self.deletion_rate, self.point_rate])
        probabilities = rates/sum(rates)
        choice = np.random.choice([0,1,2], 1, p = probabilities)
        return events[choice]

    def fixation_check(self, new_genes: np.ndarray[list[float]]):
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

    def fixation(self,new_genes : np.ndarray[list[float]], display : bool = True) -> None:
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
        self.memory.append(self.current_pos)
        self.fitness.append(self.current_fitness) 
        self.nb_genes.append(len(new_genes))
        
        if display:
            print(f'\nNew genome fixated with fitness {self.current_fitness}')

        return


    def historicity_test(self):
        """
        Resumes Evolution after the 10**5 generations to see if there are differences in
        the evolutionnary path taken

        ------
        Parameters : 
            All necessary parameters are self defined in the class object (see __init__) : 
            fitness, genes and current_pos
        
        ------
        Return : 
            None 
            Call for evolve_successive again, but with 10**5 less generations 
        
        """
        self.fitness = self.memory_fitness
        self.genes = self.memory_genes
        self.current_pos = self.memory_position
        self.current_fitness = self.fitness_calc(self.current_pos)
        self.evolve_successive(3*10**5, "always_all_gene")

    def plot_historic_fitness(self, fitness1, fitness2):
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

    def ploting_results(self, fitness, effects, time):
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
        plt.figure()

        plt.subplot(1, 2, 1)
        plt.plot(fitness)
        plt.xlabel('Time')
        plt.ylabel('Fitness')
        plt.title('Evolution of Fitness Over Time')

        plt.subplot(1, 2, 2)
        effects = np.array(effects)
        effects = effects[effects>=0] # comment this lign if you also want to see deleterious effect
        plt.plot(effects, '.', markersize=3)
        plt.hlines(0, 0 , len(effects), "k", "--")
        plt.xlabel('Time')
        plt.ylabel('Fitness Effect (s)')
        plt.title('Fitness Effects of Mutations Over Time')

        plt.show()

    def ploting_path(self, memory):
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
        path = []
        for pos in memory :
            path.append(np.linalg.norm(pos)) # compute the distance to the origin for every position 

        plt.figure()
        plt.plot(path)
        plt.xlabel('Number of Fixed Mutation')
        plt.ylabel('Distance to the optimum')
        plt.title('Evolution of the Distance to the Optimum of Phenotype after each Fixed Mutation')

        plt.show()
    
    def plot_vizualised_path(self):
        """
        Plots the final phenotype and underlying genotype of a 2- or 3-dimensional FisherGeometricModel

        Parameters
        -----
        None
        
        Returns
        -----
        None
            Plots path
        """

        fig = plt.figure(figsize = (16,9))
        if self.dimension == 2:
            ax = fig.add_subplot()
            for radius in [2,5,10,15,20,30]:
                circ = plt.Circle((0,0),radius, fill = False, color = 'k')
                ax.add_patch(circ)
            kwargs = {'color': 'k'}

        elif self.dimension == 3:
            ax = fig.add_subplot(projection = '3d')
            kwargs = {'scale_units' : 'xy', 'angles' : 'xy', 'color': 'k'}
        else:
            raise Warning("Unable to plot a 2D graph when dimensions > 3")
        
        number_of_genes = np.shape(self.genes)[0]
        genome = np.resize(self.genes.copy(),(number_of_genes,self.dimension))

        position = np.array(self.init_pos)
        ax.scatter(*position, c = 'r')
        for gene in genome:
            ax.quiver(*position,*gene, **kwargs)
            position += gene
            ax.scatter(*position, c = 'g')
        
        origin = self.optimum

        ax.scatter(*origin, c = 'k')        
        ax.grid()
        plt.show()

    def ploting_size(self, nb_genes, mean_size, std_size):
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
        ci = [1.96*std_size[k]/np.sqrt(nb_genes[k*3]) for k in range(len(std_size))]
        list_lower = [mean_size[i] - ci[i] for i in range(len(ci))]
        list_upper = [mean_size[i] + ci[i] for i in range(len(ci))]

        x = np.arange(0, len(ci), 1) # abscisse for the confidence intervals to be plotted

        plt.figure()

        plt.subplot(1, 2, 1)
        plt.plot(nb_genes)
        plt.xlabel('Mutational Events')
        plt.ylabel('Number of genes in the genotype')
        plt.title('Evolution of the Number of genes with Time')

        ax = plt.subplot(1, 2, 2)
        ax.plot(mean_size)
        ax.fill_between(x, list_lower, list_upper, alpha = .1)
        plt.xlabel('Time')
        plt.ylabel('Mean size of genes in the genotype')
        plt.title('Evolution of the size of genes with Time')

        plt.show()

def mean_list(lists):
    """
    Compute the mean list of a set of lists, where we want the mean of values group by indexes

    ------
    Parameters :
        lists : list
        List of lists on which we want to compute the mean operation

    ------
    Return : 
        mean_values : list
        List having at position i the mean of the values of index i in every lists

    """
    transposed_lists = list(zip(*lists)) # group the element of each list having the same index.
    # Calculate the mean for each group of elements
    mean_values = [sum(group) / len(group) for group in transposed_lists]
    return mean_values # return the mean list having at position i the mean of the values of index i in every lists

def standard_deviation(lists):
    """
    Compute the standard deviation list of a set of lists, where we want the std of values group by indexes

    ------
    Parameters :
        lists : list
        List of lists on which we want to compute the mean operation

    ------
    Return : 
        std_values : list
        List having at position i the std of the values of index i in every lists
    """
    transposed_lists = list(zip(*lists)) # group the element of each list having the same index.
    # Calculate the std for each group of elements
    std_values = [np.std(group) for group in transposed_lists]
    return std_values # return the std list having at position i the std of the values of index i in every lists

def simulation(fgm : FisherGeometricModel, n_generations):
    """
    Simulate the evolution of an individual thanks to the given parameters. 
    Plot evolutionary path using the evolution of fitness, position and number of genes over times, 
    but also the fitness effect of each mutational event, beneficial (or not).

    ------
    Parameters : 
        n_traits : int
        Number of phenotypic traits defining the dimension of the space
        initial_position : np.array
        One dimensional Numpy array of size n_traits representing the position of the initial phenotype of the individual
        alpha : float
        Robustness parameter in fitness function. Here, alpha always equals 1/2
        Q : int
        Epistasis parameter in fitness function. Here, Q always equals 2
        sigma_mut : float
        standard deviation of the mutation's distribution
        duplication_rate : float
        rate of duplication in terms of number of duplication per gene and generations
        deletion_rate : float
        rate of deletion in terms of number of deletion per gene and generations
        point_rate : float
        rate of mutation in terms of number of mutation par gene and generations
        ratio : float
        ratio between the standard deviation (size) of the first gene and the one of mutations
        method : str
        method used to create the first gene. Can be : random, parallel, orthogonal, semi_neutral, neutral, only_one_deleterious_direction
        mutation_method : str
        method used to make mutations at each generations : can be one_gene, all_gene, always_all_gene. The most used method here
        is the last as we mutate each genes at each generations.
        n_generations : int
        number of time step to simulate in the evolution. 

    ------
    Return :
        None
        plot graphics : evolution of fitness over time / distance to the optimum over time / 
        number of genes over time / fitness effect of mutational events
        print the last distance to the optimum reached

    """
    # Simulation
    print(fgm.methods) # It seems like the closer we are to the optimum, the lesser there are dupl and del. (and even mutation)
    fgm.ploting_results(fgm.fitness, fgm.effects, n_generations)
    fgm.ploting_path(fgm.memory)
    print(fgm.genes)
    # fgm.ploting_size(fgm.nb_genes, fgm.mean_size, fgm.std_size) # le nombre de gènes augmentent très vite au début (les duplciations sont fréquentes) puis ce stabilise jusqu'à la fin
    print(np.linalg.norm(fgm.memory[-1]))

def historic_simulation(n_traits, initial_position, alpha, Q, sigma_mut, duplication_rate, deletion_rate, point_rate, ratio, method, mutation_method, n_generations):
    """
    Simulate the evolution of an individual thanks to the given parameters. 
    Remember the position and genotype of the individual after 100000 generations. 
    Resumes a new evolutionnary path at this position to see the possible 
    differences of dynamic in GFGM when we begin at a previous point. 

    ------
    Parameters : 
        n_traits : int
        Number of phenotypic traits defining the dimension of the space
        initial_position : np.array
        One dimensional Numpy array of size n_traits representing the position of the initial phenotype of the individual
        alpha : float
        Robustness parameter in fitness function. Here, alpha always equals 1/2
        Q : int
        Epistasis parameter in fitness function. Here, Q always equals 2
        sigma_mut : float
        standard deviation of the mutation's distribution
        duplication_rate : float
        rate of duplication in terms of number of duplication per gene and generations
        deletion_rate : float
        rate of deletion in terms of number of deletion per gene and generations
        point_rate : float
        rate of mutation in terms of number of mutation par gene and generations
        ratio : float
        ratio between the standard deviation (size) of the first gene and the one of mutations
        method : str
        method used to create the first gene. Can be : random, parallel, orthogonal, semi_neutral, neutral, only_one_deleterious_direction
        mutation_method : str
        method used to make mutations at each generations : can be one_gene, all_gene, always_all_gene. The most used method here
        is the last as we mutate each genes at each generations.
        n_generations : int
        number of time step to simulate in the evolution. 

    ------
    Return :
        None
        plot graphics : evolution of fitness over time for both path

    """
    # Simulation
    fgm = FisherGeometricModel(n_traits, initial_position, alpha, Q, sigma_mut, duplication_rate, deletion_rate, point_rate, ratio, method)
    fgm.evolve_successive(n_generations, mutation_method)
    fitness1 = fgm.fitness.copy()
    fgm.historicity_test()
    fitness2 = fgm.fitness
    fgm.plot_historic_fitness(fitness1, fitness2)



####################
#### Parameters ####
####################p
if __name__ == "__main__" :
    n_traits = 2  # Number of traits in the phenotype space n
    # initial_position = np.ones(n_traits)*5/np.sqrt(n_traits) # Quand la position initiale est plus éloigné de l'origine, la pop à bcp moins de mal à s'améliorer (et les mutations sont plus grandes ?)
    # problème : peut pas partir de très loin : si on augmente trop la position initial ça fait des divisions par 0 dans le log et plus rien ne marche

    # initial_position = np.zeros(n_traits)
    # initial_position[0] = 25 # Initial phenotype on an axe

    d = 20 # Wanted initial distance to the optimum
    initial_position = np.random.normal(0, 1, n_traits)
    initial_position /= np.linalg.norm(initial_position)
    initial_position *= d
    r = 0.5
    n_generations = 1*10**4  # Number of generations to simulate
    # sigma_mut = r/np.sqrt(n_traits) # Standard deviation of the mutation effect size # Tenaillon 2014
    sigma_mut = 0.01 # énormement de duplication/deletion par rapport au nombre de mutation quand on baisse sigma (voir sigma=0.01)
    # here sigma is the same on every dimension
    alpha = 1/2
    Q = 2
    point_rate = 1e-4 # rate of mutation mu
    duplication_rate = 1e-2 # /gene/generation
    deletion_rate = 1e-2 # /gene/generation
    ratio = 5 # ratio between sigma_gene and sigma_mut (size of the first gene) == importance of duplication versus mutation

    initial_gene_method = "random"
    mutation_method = "always_all_gene"
    fgm = FisherGeometricModel(n_traits, initial_position, alpha, Q, sigma_mut, duplication_rate, deletion_rate, point_rate, ratio, initial_gene_method)
    fgm.evolve_successive(n_generations, mutation_method)
    print(f"Number of [duplications, deletions, point mutations] = {np.sum(fgm.methods,axis = 0)}")
    fgm.plot_vizualised_path()
    # Random simulation :
    # simulation(fgm, n_generations)
    

    # Historicity test
    # historic_simulation(n_traits, initial_position, alpha, Q, sigma_mut, duplication_rate, deletion_rate, point_rate, ratio, method, mutation_method, n_generations)