import numpy as np
import matplotlib.pyplot as plt
import cProfile

class FisherGeometricModel() :
    def __init__(self, n, initial_position, size, alpha, Q, sigma_mut, duplication_rate, deletion_rate, mutation_rate, ratio) :
        """
        Initialized the main parameters and variables used in our modified FGM.

        ------
        Parameters : 
            see comment on each line

        ------ 
        Return :
            None
            (memorize paramters and variables for later use in other methods)
            
        """
        self.dimension = n # dimension of the phenotypic space = number of main indepedent phenotypic traits that are used as axis is the model
        self.N = size # size of the population studied
        self.optimum = np.zeros(n) # position of the optimum. For simplicity, it has been put at the origin as in many other studies using FGM
        self.alpha = alpha # robustness parameter 
        self.Q = Q # epistasis parameter 
        # alpha and Q influence the decay rate and the curvature of the fitness function (Tenaillon 2014)
        self.sigma_mut = sigma_mut # standart deviation of the mutational vector on each axis

        self.init_pos = initial_position # initial phenotype/genotype = position in the space of the individual.
        self.genes = [self.create_random_first_gene(ratio)] # random first gene in the genotype studied, it must be at least neutral to be kept.
        # ratio = ratio between sigma_gene and sigma_mut (size of the first gene) == importance of duplication versus mutation
        # self.genes = [self.create_fixed_first_gene(ratio, "orthogonal")] # chose the direction/size of the first gene 

        self.final_pos = initial_position + np.sum(self.genes, axis=0) # the real phenotypic position of the individual is computed by adding the genes vectors to the initial position
        self.initial_fitness = self.fitness_function(self.final_pos) # Compute the fitness of the actual phenotype before any mutation happen

        self.duplication_rate = duplication_rate 
        self.deletion_rate = deletion_rate
        self.mutation_rate = mutation_rate
        # rate of duplication and deletion in nb/gene/generation and rate of mutation in nb/genome/generation

        self.memory = [self.init_pos, self.final_pos] # memorize the to first position, which are the initial phenotype of the individual and its phenotype after addition of the first gene
        self.fitness = [] # list that will memorize the fitness values of the phenotype at each iteration
        self.effects = [] # list that will memorize the effects values of the change in phenotype/genotype at each iteration
        self.methods = [] # # list that will memorize the method use to modificate the genotype at each iteration
        self.nb_genes = [1] # at the beginning, the individual only has one gene in the genome

    def create_random_first_gene(self, r):
        """
        Randomly draw a n-dimensional vector from the same normal distribution as for mutation, 
        until the vector, representing a gene, is beneficial (moving from the initial position in the direction of the gene
        improve fitness)

        ------
        Parameters :
            r : float
            Parameter which allow to play on the ratio of the standart deviation of the size of a gene and a mutation 
            Other parameters arre self defined in the class object (see __init__). Useful parameters are init_pos, sigma_mut,
            dimension.

        ------
        Return : 
            gene : np.darray
            1-dimensional numpy array of size n (dimension of the phenotypic space) representing the gene.
 
        """
        s = -1 
        while s < 0 : # we consider that the first gene must be at least neutral or beneficial so that the organism survive
            gene = np.random.normal(0, r*self.sigma_mut, self.dimension) # We draw the gene as if it was a mutation of the starting point in the Standart FGM
            new_pos = self.init_pos + gene # compute the new phenotypic position after adding the gene 
            s = self.fitness_effect(self.fitness_function(self.init_pos), self.fitness_function(new_pos)) # compute the fitness effect of such a gene.
        # print(np.linalg.norm(gene))
        return gene
    
    def create_fixed_first_gene(self, r, direction):
        if direction == "parallel" : 
            return np.ones(self.dimension) * (-r * self.sigma_mut) # gene on the axe to the optimum (for the chosen init_pos) : in this case the gene is optimal for adaptation, the simulation only make duplication 
        elif direction == "orthogonal" :
            gene = np.ones(self.dimension) * (-r * self.sigma_mut)
            gene[0] *= 0 # 0 si on part sur un axe (mettre toutes les valeurs à 0 sauf 1 dans init_pos). Dans ce cas le gène est délétère : il est supprimer et rien ne peut se faire de mieux ensuite.
            # si on utilise cette méthode sans partir sur un axe, le gène est quasiment bien aligné, dans ce cas c'est un peu comme parallele : bcp de duplication au début, mais ensuite mute un peu et supprime les gènes mauvais ?
            return gene 
        elif direction == "only_one_deleterious_direction" :
            gene = np.ones(self.dimension) * (-r * self.sigma_mut)
            gene[0] *= -1
            return gene # mute a little to correct direction, then makes a lot of duplication, then delete some genes (+mute and duplicate) == get ride of the bad genes, then mute a lot
    

    def mutation_on_one_gene(self, list_genes):
        """
        Randomly mutate some genes in the genotype by adding a gaussian noise to them. The number of
        genes to mutate is drawn from a poisson distribution of parameter the mutation rate.

        ------
        Parameters :
            list_genes : list
            List of 1-D numpy arrays representing genes. It may differ from self.genes becasue some might have been duplicated or deleted. 
            Other parameters are self defined in the class object (see __init__). The useful parameters here are 
            N, dimension, sigma_mut, mutation_rate.

        ------
        Return : 
            list_genes : list
            List of 1-dimensional numpy array of size n (dimension of the phenotypic space) representing the genes
            after mutation.
            nb_mut > 0 : boolean
            Becomes True if there is at least one mutation

        """
        nb_mut = np.random.poisson(self.mutation_rate * self.N) # Blanquart 2014 ; number of mutation to do. the rate is multiply by the size of the population so that it represent the number of mutation per generation.
        
        list_genes = np.array(list_genes) # convert the list of genes into a numpy array to allow operation on arrays.
        if nb_mut > 0: 
            indices = np.random.randint(0, len(list_genes), nb_mut) # randomly choose the genes to mutate
            mutations = np.random.normal(0, self.sigma_mut, (nb_mut, self.dimension)) # Tenaillon 2014, Blanquart 2014 ; draw the mutation from a normal distribution of variance sigma_mut**2
            list_genes[indices] = list_genes[indices] + mutations # modify the corresponding genes by adding the mutation vector to them.

        return list_genes.tolist(), nb_mut > 0 # convert the list of genes back to a list (necessary because some operations use in the class only work on lists)
    
    def mutation_on_every_genes(self, list_genes):
        """
        Randomly mutate every genes in the genotype by adding a gaussian noise to them. The mutation
        may differ from one gene to another. The number of mutation is drawn from a poisson distribution 
        of parameter the mutation rate.

        ------
        Parameters :
            list_genes : list
            List of 1-D numpy arrays representing genes. It may differ from self.genes becasue some might have been duplicated or deleted. 
            Other parameters are self defined in the class object (see __init__). The useful parameters here are 
            N, dimension, sigma_mut, mutation_rate.

        ------
        Return : 
            list_genes : list
            List of 1-dimensional numpy array of size n (dimension of the phenotypic space) representing the genes
            after mutation.
            nb_mut > 0 : boolean
            Becomes True if there is at least one mutation

        """
        nb_mut = np.random.poisson(self.mutation_rate*self.N) # Blanquart 2014 ; number of mutation to do. the rate is multiply by the size of the population so that it represent the number of mutation per generation.
        n = len(list_genes)
        if nb_mut > 0 : 
            for i in range(nb_mut):
                m = np.random.normal(0, np.sqrt(n)*self.sigma_mut, size=(n, self.dimension)) # draw the mutation from a normal distribution of variance n*sigma_mut**2 (variance of a sum of mutation)
                list_genes = [list_genes[i] + m[i] for i in range(n)] # modify every genes in the list by adding the corresponding mutation. All genes do not mutate the same way

        return list_genes, nb_mut > 0
    
    def duplication(self):
        """
        Duplicate genes from the list of vectors of genes if necessary.
        The number of duplication to do is drawn from a poisson distribution having the duplication rate as parameter

        ------
        Parameters :
            Every parameters are self defined in the class object (see __init__)
            The useful paramaters here are genes, N, duplication_rate.

        ------
        Return
            list_genes : list
            list of 1-D numpy arrays (genes) modified after duplication of some genes.
            nb_dupl > 0 : boolean
            True if a duplication have been made.

        """
        n = len(self.genes)
        list_genes = self.genes.copy() # make a copy of the initial list of genes so that it is not changed if the modification are not fixed afterward
        nb_dupl = np.random.poisson(self.duplication_rate*n*self.N) # number of duplication to do. the rate is multiply by the number of gene and the size of the population so that it represent the number of duplication per generation.
        if nb_dupl > 0 :
            indices = np.random.randint(0, n, min(nb_dupl, n)) # randomly choose the genes to duplicate, if there are more duplication to do than genes, just duplicate all genes
            for index in indices :
                    list_genes.append(self.genes[index]) # copy the duplicated genes in the list
        return list_genes, nb_dupl > 0
    
    def deletion(self):
        """
        Delete genes from the list of vectors of genes if necessary.
        The number of deletion to do is drawn from a poisson distribution having the deletion rate as parameter

        ------
        Parameters :
            Every parameters are self defined in the class object (see __init__)
            The useful paramaters here are genes, N, deletion_rate.

        ------
        Return
            list_genes : list
            list of 1-D numpy arrays (genes) modified after deletion of some genes.
            nb_del > 0 : boolean
            True if a deletion have been made.

        """
        n = len(self.genes)
        list_genes = self.genes.copy() # make a copy of the initial list of genes so that it is not changed if the modification are not fixed afterward
        nb_del = np.random.poisson(self.deletion_rate*n*self.N) # number of deletion to do. the rate is multiply by the number of gene and the size of the population so that it represent the number of deletion per generation.
        for i in range(nb_del):
            if len(list_genes) > 0 :
                index = np.random.randint(0, len(list_genes)) # randomly choose the gene to delete
                list_genes.pop(index) # suppress the deleted gene from the list

        return list_genes, nb_del > 0
    
    def duplication_deletion(self): # 4 sec
        """
        Duplicate and delete some genes from the list of vectors of genes if necessary.
        The numbers of duplication and deletion to do are drawn from poisson distribution having the corresponding rate as parameter
        There is a 50/50 percent chance that duplications happen before deletion and inversely.

        ------
        Parameters :
            Every parameters are self defined in the class object (see __init__)
            The useful paramaters here are genes, N, duplication and deletion rate.

        ------
        Return
            list_genes : list
            list of 1-D numpy arrays (genes) modified after duplication and deletion of some genes.
            dupl : boolean
            if dupl is True, a duplication have been made.
            dele : boolean
            if dele is True, a deletion have been made.

        """
        list_genes = self.genes.copy() # make a copy of the initial list of genes so that it is not changed if the modification are not fixed afterward
        dupl = False
        dele = False
        if np.random.rand() < 0.5 : # randomly decide about the order between duplication and deletion
            nb_dupl = np.random.poisson(self.duplication_rate*len(list_genes)*self.N) # number of duplication to do. the rate is multiply by the number of gene and the size of the population so that it represent the number of duplication per generation.
            if nb_dupl > 0:
                dupl = True # dupl is put at True if there is at least one duplication
                indices = np.random.randint(0, len(list_genes), min(nb_dupl, len(self.genes))) # randomly choose the gene to duplicate, if there are more duplication to do than genes, just duplicate all genes
                for index in indices :
                    list_genes.append(self.genes[index]) # add a copy of each gene in the list

            nb_del = np.random.poisson(self.deletion_rate*len(list_genes)*self.N) # number of deletion to do. the rate is multiply by the number of gene and the size of the population so that it represent the number of deletion per generation.
            if nb_del > 0:
                dele = True # dele is put at True if there is at least one deletion
                for i in range(nb_del):
                    if len(list_genes) > 0: # if there is at least one gene to suppress
                        index = np.random.randint(0, len(list_genes)) # randomly choose the gene to delete
                        list_genes.pop(index) # suppress it from the list of gene

        else : # same but the order it inversed
            nb_del = np.random.poisson(self.deletion_rate*len(list_genes)*self.N)
            if nb_del > 0:
                dele = True
                for i in range(nb_del):
                    if len(list_genes) > 0:
                        index = np.random.randint(0, len(list_genes))
                        list_genes.pop(index)
            
            nb_genes = len(list_genes) # we take the number of gene in the actualised list because some genes may have been suppressed from the initial one.
            nb_dupl = np.random.poisson(self.duplication_rate*nb_genes*self.N)
            if nb_dupl > 0:
                dupl = True
                indices = np.random.randint(0, len(list_genes), min(nb_dupl, len(self.genes))) 
                for index in indices :
                    list_genes.append(self.genes[index]) 

        return list_genes, dupl, dele
    
    def duplication_deletion_v2(self): # faire une deuxième version avec test par gène SVP
        """
        Duplicate and delete some genes from the list of vectors of genes if necessary.
        The number of duplication/deletion to do is drawn from a poisson distribution having the rearrangement rate as parameter
        Then, for each rearrangement, there is a 50/50 chance that it is a duplication or a deletion of a random gene.

        ------
        Parameters :
            Every parameters are self defined in the class object (see __init__)
            The useful paramaters here are genes, N, duplication and deletion rate.

        ------
        Return
            list_genes : list
            list of 1-D numpy arrays (genes) modified after duplication and deletion of some genes.
            dupl : boolean
            if dupl is True, a duplication have been made.
            dele : boolean
            if dele is True, a deletion have been made.

        """
        list_genes = self.genes.copy() # make a copy of the initial list of genes so that it is not changed if the modification are not fixed afterward
        dupl = False
        dele = False
        nb_dupl_del = np.random.poisson(2*self.duplication_rate*len(list_genes)*self.N) # number of rearrangement to do. the rate is multiply by the number of gene and the size of the population so that it represent the number of rearrangment per generation.
        for i in range(nb_dupl_del):
            if np.random.rand() < 0.5 : # randomly choose between duplication and deletion
                dupl = True # dupl is put at True if there is at least one duplication
                index = np.random.randint(0, len(self.genes)) # randomly choose a gene to copy
                list_genes.append(self.genes[index]) # add a copy of the gene to the list

            else : 
                if len(list_genes) > 0: # only possible if there actually is one gene to delete
                    dele = True # dele is put at True if there is at least one deletion
                    index = np.random.randint(0, len(list_genes)) # randomly choose a gene to suppress
                    list_genes.pop(index) # suppress this gene from the list

        return list_genes, dupl, dele

    def fitness_function(self, position): # 4sec
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
    
    def fitness_effect(self, initial_fitness, new_fitness):
        """
        Compute the fitness effect of a change in the phenotype/genotype.
        It is just the logarithm of the ratio between the fitness of the new position in the space 
        and the fitness of the position before the modification. (ancestral position).

        ------
        Parameters :
            initial_fitness : float
            fitness of the ancestral phenotype/genotype.
            new_fitness : float
            fitness of the new phenotype/genotype.

        ------
        Return
            np.log(new_fitness/initial_fitness) : float
            Fitness effect s of a modification in the phenotype/genotype of an individual (due to 
            mutation, duplication, deletion)

        """
        return np.log(new_fitness/initial_fitness) # when the new fitness is higher than the ancestral one, the ratio is > 1, so the effect is > 0
        # return new_fitness/initial_fitness - 1 # Martin 2006 (for s small, not used here)
        # if > 0, the mutation as improve the fitness, it is beneficial
    
    def fixation_probability(self, s) :
        """
        Compute the fixation probability of a given modification in the genome, 
        having a fitness effect s.
        The computation is not the same depending on the value of s :
        if s > 0 or nearly neutral, the probability is compute using Barrett's equation
        If s is nearly 0 (neutral mutation), the probability is just 1/N (with N the size of the population).
        if s << 0 : the mutation is deleterious, it will not be fixed. 

        The possibility of nearly neutral and neutral mutation of being fixed is du to genetic drift 
        which can have antagonistic results with selection. This process have an important effect in
        little population (bottleneck and funder effect), but is negligeable compared to selection in larger population.
        That's why N always addapt the value of the denominator. 

        ------
        Parameters :
            s : float 
            fitness effect of the modification (see function fitness_effect())
            Other parameters are self defined in the class object (see __init__) : the useful parameter 
            here are N.

        ------
        Return
            p : float
            Probability of fixation of a mutation/duplication/deletion. 
            
        """
        # We consider a population with Equal Sex Ratios and Random Mating : Ne = N
        # p = 2*s # Haldane 1927 (only viable for very little s, that's why we don't use it here)
        if 100*np.abs(s) < 1/self.N : # |s| << 1/N : neutral mutation
            p = 1/self.N # p rapidly falls to zero when N becomes larger
        elif np.abs(s) < 1/(2*self.N) or s > 0 : # nearly neutral mutation
            p = 1 - np.exp(-2*s) / (1 - np.exp(-2*self.N*s)) # Barrett 2006 (in a N=1 population, beneficial mutation are necessarly fixed)
        else : # deleterious mutation
            p = 0 # deleterious mutation do not fix
        return p

    def evolve(self, time_step) : # 20 sec
        """
        Main method that simulate the evolution for a certain time. 
        At each iteration, test if there are duplications/deletions and mutations to do, 
        modify the list of genes in consequence, compute the new phenotype and its fitness, 
        test if this modification in the genome is fixed and memorize some importants caracteristics 
        (fitness, fitness effect, position, ...) that will allow us to study the simulated evolution.

        ------
        Parameters :
            time_step : int
            Number of time steps (generations) on which we want to iterate in the simulation
            Other parameters are self defined in the class object (see __init__). The useful parameters here 
            are init_pos, final_pos, genes

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
        for i in range(time_step):
            # initial_fitness = self.fitness_function(self.final_pos) # Compute the fitness of the actual phenotype before any mutation happen

            list_genes, dupl, dele = self.duplication_deletion() # test if there are some duplications and deletions to do
            # list_genes, dupl, dele = self.duplication_deletion_v2() # test if there are some duplication and/or deletion to do (not exactly the same way of computating it)
            
            if len(list_genes) > 0 : # mutation possible only if there is at least one gene (could be 0 gene after deletion)
                # If mutation on one gene :
                list_genes, mut = self.mutation_on_one_gene(list_genes) # test if there are some mutations to do ; mutation on one or some gene
        
                # If mutation on every genes : 
                # list_genes, mut = self.mutation_on_every_genes(list_genes) # mutation on every genes
            
            if dupl or dele or mut : # if there was at least one modification in the genome
                new_final_pos = self.init_pos + np.sum(list_genes, axis=0) # compute new phenotypic position

                new_fitness  = self.fitness_function(new_final_pos) # its new fitness
                s = self.fitness_effect(self.initial_fitness, new_fitness) # the fitness effect of the modification
                pf = self.fixation_probability(s) # its fixation probality depending on its fitness effect

                if np.random.rand() < pf : # the mutation is fixed
                    method = []
                    if dupl :
                        method.append("duplication")
                    if dele :
                        method.append("deletion")
                    if mut :
                        method.append("mutation")
                    self.methods.append(method) # remember the method(s) that created this modification
                    
                    self.genes = list_genes # actualize the genome
                    self.final_pos = new_final_pos # and the phenotype
                    self.initial_fitness = new_fitness # and the fitness of the phenotype
                    self.memory.append(new_final_pos)
                    self.fitness.append(new_fitness) 

                else : # if the modification didn't fixed 
                    self.fitness.append(self.initial_fitness)

                self.effects.append(s)

            else : # if nothing happen
                self.fitness.append(self.initial_fitness) 
                self.effects.append(0)

            self.nb_genes.append(len(self.genes)) # remember the number of genes in the genotype after this iteration
                
        # return self.memory, self.fitness, self.effects, self.methods, self.nb_genes
    
    def evolve_successive(self, time_step) : # 20 sec
        """
        Main method that simulate the evolution for a certain time. 
        At each iteration, only make one kind of change in the genome (duplication, deletion, mutation), 
        modify the list of genes in consequence, compute the new phenotype and its fitness, 
        test if this modification in the genome is fixed and memorize some importants caracteristics 
        (fitness, fitness effect, position, ...) that will allow us to study the simulated evolution.

        ------
        Parameters :
            time_step : int
            Number of time steps (successive mutations = 3*time_step) on which we want to iterate in the simulation
            Other parameters are self defined in the class object (see __init__). The useful parameters here 
            are init_pos, final_pos, genes

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
        for i in range(time_step):
            if np.random.rand() < 0.5 :
                list_genes, dupl = self.duplication() # test if there are some duplications to do
                
                if dupl :
                    self.test_fixation(list_genes, "duplication")

                list_genes, dele = self.deletion() # test if there are some deletions to do

                if dele : 
                    self.test_fixation(list_genes, "deletion")
            else : 
                list_genes, dele = self.deletion() # test if there are some deletions to do

                if dele : 
                    self.test_fixation(list_genes, "deletion")

                list_genes, dupl = self.duplication() # test if there are some duplications to do
                
                if dupl :
                    self.test_fixation(list_genes, "duplication")
            
            if len(self.genes) > 0 :
                list_genes, mut = self.mutation_on_every_genes(self.genes) # test if there are some mutations to do

                if mut :
                    self.test_fixation(list_genes, "mutation")
                     
        # return memory, fitness, effects, methods, nb_genes
    
    def test_fixation(self, list_genes, method):
        """
        Test if a change in the genotype will be fixed. If it is, actualized every parameters impacted.
        This method is used in the evolve_succesive simulation.

        ------
        Parameters :
            list_genes : list
            The list of genes of the genotype after the mutation/duplication/deletion 
            method : str
            The method used to make the change in the genotype

        ------
        Return :
            None
            Only actualized self parameters for next iterations

        """
        new_final_pos = self.init_pos + np.sum(list_genes, axis=0) # compute new phenotypic position

        new_fitness  = self.fitness_function(new_final_pos) # its new fitness
        s = self.fitness_effect(self.initial_fitness, new_fitness) # the fitness effect of the modification
        pf = self.fixation_probability(s) # its fixation probality depending on its fitness effect

        if np.random.rand() < pf : # the mutation is fixed
            self.genes = list_genes # actualize the genome
            self.final_pos = new_final_pos # and the phenotype
            self.initial_fitness = new_fitness
            self.memory.append(new_final_pos)
            self.fitness.append(new_fitness) 
            self.methods.append(method) # remember the method that created this modification

        else : # if the modification didn't fixed 
            self.fitness.append(self.initial_fitness)

        self.effects.append(s)
        self.nb_genes.append(len(self.genes)) # remember the number of genes in the genotype after this iteration

    def ploting_results(self, fitness, effects, time):
        """
        Plot two important graphs that allow us to follow the evolution of fitness other time
        and the different value of the fitness effect s other time.

        ------
        Parameters :
            fitness : list
            List of the fitness (float) of the individual at each time_step
            effects : list
            List of the fitness effect (float) of the modification of the phenotype/genotype 
            that happend at each time step.
            time : int
            Number of time_step used in the simulation

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
            List of position (vectors) in the phenotypic space after each fixed mutations.

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
    
    def ploting_size(self, nb_genes):
        """
        Plot the evolution of the number of genes (due to duplication and deletion) 
        with respect to time.

        ------
        Parameters :
            nb_genes : list
            List of the number of genes in the genome of the individual at each time step

        ------
        Return
            None
            (show the graph in an other window)
            
        """
        plt.figure()
        plt.plot(nb_genes)
        plt.xlabel('Time')
        plt.ylabel('Number of genes in the genotype')
        plt.title('Evolution of the Number of genes with Time')

        plt.show()


def main():
    # Parameters
    n_traits = 50  # Number of traits in the phenotype space n
    initial_position = np.ones(n_traits)*25/np.sqrt(n_traits) # Quand la position initiale est plus éloigné de l'origine, la pop à bcp moins de mal à s'améliorer (et les mutations sont plus grandes ?)
    # problème : peut pas partir de très loin : si on augmente trop la position initial ça fait des divisions par 0 dans le log et plus rien ne marche
    
    # initial_position = np.zeros(n_traits)
    # initial_position[0] = 25 # pour partir sur un axe
    
    n_generations = 5*10**5  # Number of generations to simulate (pas vraiment, voir commentaire sur Nu)
    r = 0.5 
    sigma_mut = r/np.sqrt(n_traits) # Standard deviation of the mutation effect size # Tenaillon 2014
    # here sigma is the same on every dimension
    population_size = 10**3 # Effective population size N
    alpha = 1/2
    Q = 2
    mutation_rate = 10**(-4) # rate of mutation mu
    # La simulation actuelle à donc une echelle de temps en (Nu)**(-1) soit une mutation toute les 100 générations
    duplication = 0.2 # %
    duplication_rate = 10**(-4) # /gene/generation
    deletion =  0.1 # %
    deletion_rate = 10**(-4) # /gene/generation
    mutation = 0.7 # % (if mutation at 100 and the other at 0, same as standart FGM but with genes instead of mutations)
    ratio = 3 # ratio between sigma_gene and sigma_mut (size of the first gene) == importance of duplication versus mutation
    # etrangement le nombre final de duplication est plus élevé avec ratio = 0.5 que avec 3 + plus de duplication au départ ??

    # Simulation
    fgm = FisherGeometricModel(n_traits, initial_position, population_size, alpha, Q, sigma_mut, duplication_rate, deletion_rate, mutation_rate, ratio)
    fgm.evolve_successive(n_generations)
    fgm.ploting_results(fgm.fitness, fgm.effects, n_generations)
    fgm.ploting_path(fgm.memory)
    fgm.ploting_size(fgm.nb_genes) # le nombre de gènes augmentent très vite au début (les duplciations sont fréquentes) puis ce stabilise jusqu'à la fin
    print(fgm.methods) # It seems like the closer we are to the optimum, the lesser there are dupl and del. (and even mutation)
    print(np.linalg.norm(fgm.memory[-1]))

# cProfile.run('main()', sort="tottime") # 15 sec --> 20 sec (ajout)
main()

"""# complexity of the phenotypic space
list_n = [2, 5, 10, 20, 30, 50]
results = {}
for n in list_n:
    initial_position = np.ones(n)*10/np.sqrt(n) 
    fgm = FisherGeometricModel(n, initial_position, population_size, alpha, Q, sigma_mut, duplication_rate, deletion_rate, mutation_rate, ratio)
    memory, fitness, effects, methods, nb_genes = fgm.evolve(n_generations)  
    results[n] = fitness

plt.figure()
for n, fitness in results.items():
    plt.plot(fitness, label=f'n_traits = {n}')

plt.xlabel('Time')
plt.ylabel('Fitness')
plt.title('Evolution of Fitness Over Time with Different Numbers of Traits')
plt.legend()
plt.show()"""

'''# cost of complexity :
results = {}
count = 0
for i in range(100):
    fgm = FisherGeometricModel(n_traits, initial_position, population_size, alpha, Q, sigma_mut, duplication_rate, deletion_rate, mutation_rate, ratio)
    memory, fitness, effects, methods, nb_genes = fgm.evolve(n_generations)  
    results[fitness[-1]] = len(fgm.genes)
    count += 1
    print(count)

    """n = len(fgm.genes)
    if n in results.keys():
        results[n].append(fitness[-1])
    else :
        results[n] = fitness[-1]"""

final_fitness = list(results.keys())
nb_genes = list(results.values())
plt.plot(nb_genes, final_fitness, "o")
plt.xlabel('Number of genes')
plt.ylabel('Final Fitness')
plt.title('Final fitness of the population depending on the number of genes it has')
plt.show()
print(results)'''
# The more the genes are duplicated, the more the final fitness is far from optimum (fopt=1)


"""# Change in init pos
n_traits = 50
list_init = [np.ones(n_traits)*0.5/np.sqrt(n_traits), 
             np.ones(n_traits)*1/np.sqrt(n_traits), 
             np.ones(n_traits)*5/np.sqrt(n_traits),
             np.ones(n_traits)*10/np.sqrt(n_traits)]
results = {}
count = 0
for init_pos in list_init :
    fgm = FisherGeometricModel(n_traits, init_pos, population_size, alpha, Q, sigma_mut, duplication_rate, deletion_rate, mutation_rate, ratio)
    memory, fitness, effects, methods, nb_gene = fgm.evolve(n_generations)
    d = np.linalg.norm(init_pos)
    nb_mut = len(memory) - 2 # memory a les positions init et après le premeir gène donc il faut faire -2
    effects = np.asarray(effects)
    if len(effects[effects>0]) == 0 :
        mean_effect = 0
    else :
        mean_effect = np.mean(effects[effects>0])
    results[d] = (fitness, nb_mut, mean_effect)
    count += 1
    print(count)

plt.figure()
for d, val in results.items():
    plt.plot(val[0], label=f'distance_to_the_optimum = {d}')

plt.xlabel('Time')
plt.ylabel('Fitness')
plt.title('Evolution of Fitness Over Time with Different Initial Position')
plt.legend()
plt.show()
# we clearly see that when we begin closer to the optimum, it is very difficult to become even more efficient.
# interesting when mutation on one gene : the 3 last init pos finish at roughthly the same fitness : difficult to go further
plt.figure()
for d, val in results.items():
    plt.plot(d, val[1], "o")

plt.xlabel('Distance to the optimum')
plt.ylabel('Number of fixed mutations/rearrangments')
plt.title('Number of fixed change in the genotype depending on the initial positions')
plt.show()
# less visible because we only see the number of mutations and not their effect but still globally nb of mutations increase with the dist.
# the distribution is even inverse when we take dupl OR del and mutation on every gene
plt.figure()
for d, val in results.items():
    plt.plot(d, val[2], "o")

plt.xlabel('Distance to the optimum')
plt.ylabel('Mean effect of beneficial mutation')
plt.title('Mean effect of beneficial mutation depending on the initial positions')
plt.show()
# Nice but seems like the effect diminish when d=10 : when to far away its a problem to ? 
# not a problem when we take mutation on one gene only --> the graph is nice"""

"""# Role of mutation rate
initial_position = np.ones(n_traits)*5/np.sqrt(n_traits) 
list_mu = [10**(-4), 10**(-5), 10**(-6)]
results = {}
for mu in list_mu :
    fgm = FisherGeometricModel(n_traits, initial_position, population_size, alpha, Q, sigma_mut, duplication_rate, deletion_rate, mu, ratio)
    memory, fitness, effects, methods, nb_genes = fgm.evolve(n_generations)
    results[mu] = fitness

plt.figure()
for mu, fitness in results.items():
    plt.plot(fitness, label=f'mutation_rate = {mu}')

plt.xlabel('Time')
plt.ylabel('Fitness')
plt.title('Evolution of Fitness Over Time with Different mutation rate')
plt.legend()
plt.show()"""