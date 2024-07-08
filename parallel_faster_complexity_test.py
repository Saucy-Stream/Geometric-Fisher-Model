import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import time

class FisherGeometricModel() :
    def __init__(self, n, initial_position, size, alpha, Q, sigma_mut, duplication_rate, deletion_rate, mutation_rate) :
        self.dimension = n
        self.N = size
        self.optimum = np.zeros(n)
        self.alpha = alpha
        self.Q = Q
        self.sigma_mut = sigma_mut

        self.init_pos = initial_position
        self.genes = [self.create_first_gene()]

        self.final_pos = initial_position + np.sum(self.genes, axis=0)

        self.duplication_rate = duplication_rate
        self.deletion_rate = deletion_rate
        self.mutation_rate = mutation_rate
        
    def create_first_gene(self):
        s = -1 
        while s < 0 : # we consider that the first gene must be at least neutral or beneficial so that the organism survive
            gene = np.random.normal(0, self.sigma_mut, self.dimension) # on fait comme si c'était une mutation du point de départ dans le modèle FGM standart
            new_pos = self.init_pos + gene
            s = self.fitness_effect(self.fitness_function(self.init_pos), self.fitness_function(new_pos))

        return gene

    def mutation_on_one_gene(self, list_genes):
        nb_mut = np.random.poisson(self.mutation_rate * self.N)
        list_genes = np.array(list_genes)
        if nb_mut > 0:
            indices = np.random.randint(0, len(list_genes), nb_mut)
            mutations = np.random.normal(0, self.sigma_mut, (nb_mut, self.dimension))
            list_genes[indices] = list_genes[indices] + mutations
            # np.add.at(list_genes, indices, mutations)

        return list_genes.tolist(), nb_mut > 0
    
    def mutation_on_every_genes(self, list_genes):
        nb_mut = np.random.poisson(self.mutation_rate*self.N)
        n = len(list_genes)
        if nb_mut > 0 :
            for i in range(nb_mut):
                m = np.random.normal(0, np.sqrt(n)*self.sigma_mut, size=(n, self.dimension))
                list_genes = [list_genes[i] + m[i] for i in range(n)]

        return list_genes, nb_mut > 0
    
    def duplication_deletion(self): # faire une deuxième version avec test par gène SVP
        list_genes = self.genes.copy()
        dupl = False
        dele = False
        if np.random.rand() < 0.5 :
            nb_dupl = np.random.poisson(self.duplication_rate*len(list_genes)*self.N)
            indexes_dupl = set() 
            if nb_dupl > 0:
                dupl = True
                for i in range(min(nb_dupl, len(self.genes))):
                    index = np.random.randint(0, len(self.genes))
                    while index in indexes_dupl : # on considère qu'il ne peut pas y avoir deux fois une duplication sur le même gène
                        index = np.random.randint(0, len(self.genes))
                    list_genes.append(self.genes[index])
                    indexes_dupl.add(index)

            nb_del = np.random.poisson(self.deletion_rate*len(list_genes)*self.N)
            if nb_del > 0:
                dele = True
                for i in range(nb_del):
                    if len(list_genes) > 0: 
                        index = np.random.randint(0, len(list_genes))
                        list_genes.pop(index)

        else : 
            nb_del = np.random.poisson(self.deletion_rate*len(list_genes)*self.N)
            if nb_del > 0:
                dele = True
                for i in range(nb_del):
                    if len(list_genes) > 0:
                        index = np.random.randint(0, len(list_genes))
                        list_genes.pop(index)
            
            nb_genes = len(list_genes)
            # print(nb_genes)
            nb_dupl = np.random.poisson(self.duplication_rate*nb_genes*self.N)
            # print(f"2 : {nb_dupl}")
            indexes_dupl = set()
            if nb_dupl > 0:
                dupl = True
                for i in range(min(nb_dupl, nb_genes)):
                    index = np.random.randint(0, nb_genes)
                    while index in indexes_dupl :
                        index = np.random.randint(0, nb_genes)
                    list_genes.append(self.genes[index])
                    indexes_dupl.add(index)
            
        # print(len(list_genes))

        return list_genes, dupl, dele
    
    def duplication_deletion_v2(self): # faire une deuxième version avec test par gène SVP
        list_genes = self.genes.copy()
        dupl = False
        dele = False
        nb_dupl_del = np.random.poisson(2 * self.duplication_rate * len(list_genes) * self.N)
        for i in range(nb_dupl_del):
            if np.random.rand() < 0.5 :
                dupl = True
                index = np.random.randint(0, len(self.genes))
                list_genes.append(self.genes[index])

            else : 
                if len(list_genes) > 0:
                    dele = True
                    index = np.random.randint(0, len(list_genes))
                    list_genes.pop(index)

        return list_genes, dupl, dele

    def fitness_function(self, position):
        return np.exp(-self.alpha * np.linalg.norm(position)**self.Q)
    
    def fitness_effect(self, initial_fitness, new_fitness):
        return np.log(new_fitness / initial_fitness)
        # return new_fitness/initial_fitness - 1 # Martin 2006 (for s small)
        # if > 0, the mutation as improve the fitness, it is beneficial
    
    def fixation_probability(self, s) :
        # We consider a population with Equal Sex Ratios and Random Mating : Ne = N
        # p = 2*s # Haldane 1927 (only viable for very little s)
        if 100 * np.abs(s) < 1/self.N : # |s| << 1/N : neutral mutation
            p = 1/  self.N
        elif np.abs(s) < 1 / (2*self.N) or s > 0 : # nearly neutral mutation
            p = 1 - np.exp(-2*s) / (1 - np.exp(-2*self.N*s)) # Barrett 2006
        else : # deleterious mutation
            p = 0 
            # print("Deleterious")
        return p

    def evolve(self, time_step) :
        memory = [self.init_pos, self.final_pos]
        fitness = []
        effects = []
        methods = []

        for i in range(time_step):
            initial_fitness = self.fitness_function(self.final_pos)

            # list_genes, dupl, dele = self.duplication_deletion()
            list_genes, dupl, dele = self.duplication_deletion_v2()
            
            if len(list_genes) > 0 : 
                # If mutation on one gene :
                list_genes, mut = self.mutation_on_one_gene(list_genes)
        
                # If mutation on every genes : 
                # list_genes, mut = self.mutation_on_every_genes(list_genes)
            

            new_final_pos = self.init_pos + np.sum(list_genes, axis=0)
            new_fitness  = self.fitness_function(new_final_pos)
            s = self.fitness_effect(initial_fitness, new_fitness)
            pf = self.fixation_probability(s)

            if np.random.rand() < pf :
                self.genes = list_genes
                self.final_pos = new_final_pos
                memory.append(new_final_pos)
                fitness.append(new_fitness)
                method = []
                if dupl :
                    method.append("duplication")
                if dele :
                    method.append("deletion")
                if mut :
                    method.append("mutation")
                methods.append(method)

            else :
                fitness.append(initial_fitness)

            effects.append(s)
                
        return memory, fitness, effects, methods

    def ploting_results(self, fitness, effects, time):
        plt.figure()

        plt.subplot(1, 2, 1)
        plt.plot(fitness)
        plt.xlabel('Time')
        plt.ylabel('Fitness')
        plt.title('Evolution of Fitness Over Time')

        plt.subplot(1, 2, 2)
        plt.plot(effects, '.', markersize=3)
        plt.hlines(0, 0, time, "k", "--")
        plt.xlabel('Time')
        plt.ylabel('Fitness Effect (s)')
        plt.title('Fitness Effects of Mutations Over Time')

        plt.show()

    def ploting_path(self, memory):
        path = []
        for pos in memory :
            path.append(np.linalg.norm(pos))

        plt.figure()
        plt.plot(path)
        plt.xlabel('Number of Fixed Mutation')
        plt.ylabel('Distance to the optimum')
        plt.title('Evolution of the Distance to the Optimum of Phenotype after each Fixed Mutation')

        plt.show()

def run_simulation(seed, n_traits, initial_position, population_size, alpha, Q, sigma_mut, duplication_rate, deletion_rate, mutation_rate, n_generations):
    np.random.seed(seed)
    fgm = FisherGeometricModel(n_traits, initial_position, population_size, alpha, Q, sigma_mut, duplication_rate, deletion_rate, mutation_rate)
    memory, fitness, effects, methods = fgm.evolve(n_generations)
    return fitness[-1], len(fgm.genes)

if __name__ == '__main__':
    # Parameters
    n_traits = 50  # Number of traits in the phenotype space n
    initial_position = np.ones(n_traits)*10/np.sqrt(n_traits) # Quand la position initiale est plus éloigné de l'origine, la pop à bcp moins de mal à s'améliorer (et les mutations sont plus grandes ?)
    # first_gene (defini dans le init)
    n_generations = 2*10**5  # Number of generations to simulate (pas vraiment, voir commentaire sur Nu)
    r = 0.5 
    sigma_mut = r/np.sqrt(n_traits) # Standard deviation of the mutation effect size # Tenaillon 2014
    # here sigma is the same on every dimension
    population_size = 10**3 # Effective population size N
    alpha = 1/2
    Q = 2
    mutation_rate = 10**(-4) # rate of mutation mu
    duplication_rate = 10**(-4) # /gene/generation
    deletion_rate = 10**(-4) # /gene/generation

    """# Simulation
    fgm = FisherGeometricModel(n_traits, initial_position, population_size, alpha, Q, sigma_mut, duplication_rate, deletion_rate, mutation_rate)
    memory, fitness, effects, methods = fgm.evolve(n_generations)
    fgm.ploting_results(fitness, effects, n_generations)
    fgm.ploting_path(memory)"""

    # Run simulations in parallel
    n_simulations = 100
    seeds = np.random.randint(0, 2**31 - 1, n_simulations)
    t1 = time.time()
    with Pool() as pool:
        results = pool.starmap(run_simulation, [(seed, n_traits, initial_position, population_size, alpha, Q, sigma_mut, duplication_rate, deletion_rate, mutation_rate, n_generations) for seed in seeds])

    # Process results
    final_fitness, nb_genes = zip(*results)
    plt.plot(nb_genes, final_fitness, "o")
    plt.xlabel('Number of genes')
    plt.ylabel('Final Fitness')
    plt.title('Final fitness of the population depending on the number of genes it has')
    plt.show()
    t2 = time.time()
    print(t2-t1) # 322 sec = 5min20sec