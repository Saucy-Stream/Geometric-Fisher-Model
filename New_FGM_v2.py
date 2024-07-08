import numpy as np
import matplotlib.pyplot as plt

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
            init_fitness = self.fitness_function(self.init_pos)
            fitness = self.fitness_function(new_pos)
            s = self.fitness_effect(init_fitness, fitness)

        return gene

    def mutation_on_one_gene(self, list_genes):
        
        # new_genes = list_genes.copy()
        if np.random.rand() < self.mutation_rate*self.N : # enlever cette ligne si on veut faire comme dans le modèle de base une mutation par pas de temps
            index = np.random.randint(0, len(list_genes))
        
            m = np.random.normal(0, self.sigma_mut, self.dimension) # Tenaillon 2014 / Blanquart 2014
            # print("1 :", self.genes)
            # print("1:",self.genes[index])
            list_genes[index] = list_genes[index] + m
            # print("2:",self.genes[index])
            # print("2 :", self.genes)
        
        return list_genes
    
    def mutation_on_every_genes(self, list_genes):
        n = len(list_genes)
        new_genes = list_genes.copy()
        if np.random.rand() < self.mutation_rate*self.N :
            m = np.random.normal(0, np.sqrt(n)*self.sigma_mut, size=(n, self.dimension))
            new_genes = [new_genes[i] + m[i] for i in range(n)] # mutation is different on every genes

        return new_genes
    
    def duplication(self):
        n = len(self.genes)
        nb_dupl = np.random.poisson(self.duplication_rate*n*self.N)
        new_final_pos = self.final_pos.copy()
        indexes = []
        for i in range(nb_dupl):
            index = np.random.randint(0, n)
            new_final_pos += self.genes[index]
            indexes.append(index)
        return new_final_pos, indexes
    
    def deletion(self):
        n = len(self.genes)
        nb_del = np.random.poisson(self.deletion_rate*n*self.N)
        new_final_pos = self.final_pos.copy()
        indexes = []
        for i in range(nb_del):
            index = np.random.randint(0, n)
            new_final_pos -= self.genes[index]
            indexes.append(index)
        return new_final_pos, indexes
    
    def duplication_deletion(self): # faire une deuxième version avec test par gène SVP
        # print("1 :", len(self.genes))
        list_genes = self.genes.copy()
        # print("1 :", list_genes)
        p = np.random.randint(0,2)
        # print(p)
        if p == 0 :
            if np.random.rand() < self.duplication_rate*len(list_genes*self.N):
                # print("!!")
                index = np.random.randint(0, len(self.genes))
                list_genes.append(self.genes[index])
                # print("2 :", list_genes)
                    
            if np.random.rand() < self.deletion_rate*len(list_genes*self.N):
                index = np.random.randint(0, len(list_genes))
                list_genes.pop(index)
            # print("2 :", list_genes)
        else : 
            if np.random.rand() < self.deletion_rate*len(list_genes*self.N):
                index = np.random.randint(0, len(list_genes))
                list_genes.pop(index)
            
            if np.random.rand() < self.duplication_rate*len(list_genes*self.N):
                index = np.random.randint(0, len(self.genes))
                list_genes.append(self.genes[index])
        # print("2 :", len(self.genes))
        # print("3 :", len(list_genes))
        return list_genes

    def duplication_deletion_v2(self): # cas où c'est soit une dupl, soit une del, pas les deux en même temps
        list_genes = self.genes.copy()
        if np.random.rand() < 2*self.duplication_rate*len(list_genes*self.N): # dupl_rate == del_rate
            p = np.random.randint(0,2)
            if p == 0 :
                index = np.random.randint(0, len(self.genes))
                list_genes.append(self.genes[index])

            else : 
                index = np.random.randint(0, len(list_genes))
                list_genes.pop(index)

        return list_genes

    def fitness_function(self, position):
        d = np.linalg.norm(position)
        # print(d)
        w = np.exp(-self.alpha * d**self.Q)
        return w
    
    def fitness_effect(self, initial_fitness, new_fitness):
        return np.log(new_fitness/initial_fitness)
        # return new_fitness/initial_fitness - 1 # Martin 2006 (for s small)
        # if > 0, the mutation as improve the fitness, it is beneficial
    
    def fixation_probability(self, s) :
        # We consider a population with Equal Sex Ratios and Random Mating : Ne = N
        # p = 2*s # Haldane 1927 (only viable for very little s)
        if 100*np.abs(s) < 1/self.N : # |s| << 1/N : neutral mutation
            p = 1/self.N
        elif np.abs(s) < 1/(2*self.N) or s > 0 : # nearly neutral mutation
            p = 1 - np.exp(-2*s) / (1 - np.exp(-2*self.N*s)) # Barrett 2006
        else : # deleterious mutation
            p = 0 
            # print("Deleterious")
        return p

    def evolve(self, time_step) :
        memory = [self.init_pos, self.final_pos]
        fitness = []
        effects = []

        for i in range(time_step):
            # print(self.final_pos)
            # print(self.genes)
            initial_fitness = self.fitness_function(self.final_pos)
            # print(initial_fitness)
            
            # list_genes = self.duplication_deletion()
            list_genes = self.duplication_deletion_v2()
            
            # print("1:", len(list_genes))
            # print("2:", len(self.genes))
            
            # If mutation on one gene :
            if len(list_genes) > 0 : 
                list_genes = self.mutation_on_one_gene(list_genes)
            
            # If mutation on every genes :
            # if len(list_genes) > 0 :  
            #     list_genes = self.mutation_on_every_genes(list_genes)

            new_final_pos = self.init_pos + np.sum(list_genes, axis=0)
            # print(new_final_pos)

            new_fitness  = self.fitness_function(new_final_pos)
            # print(new_fitness)
            s = self.fitness_effect(initial_fitness, new_fitness)
            # print(s)
            pf = self.fixation_probability(s)

            if np.random.rand() < pf :
                # print("Youpy")
                self.genes = list_genes
                self.final_pos = new_final_pos
                memory.append(new_final_pos)
                fitness.append(new_fitness)

            else :
                fitness.append(initial_fitness)

            effects.append(s)
                
        return memory, fitness, effects

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

# Parameters
n_traits = 50  # Number of traits in the phenotype space n
initial_position = np.ones(n_traits)*5/np.sqrt(n_traits) # Quand la position initiale est plus éloigné de l'origine, la pop à bcp moins de mal à s'améliorer (et les mutations sont plus grandes ?)
n_generations = 2*10**5  # Number of generations to simulate (pas vraiment, voir commentaire sur Nu)
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

"""# Simulation
fgm = FisherGeometricModel(n_traits, initial_position, population_size, alpha, Q, sigma_mut, duplication_rate, deletion_rate, mutation_rate)
memory, fitness, effects = fgm.evolve(n_generations)
fgm.ploting_results(fitness, effects, n_generations)
fgm.ploting_path(memory)
print(len(fgm.genes))"""

"""# complexity of the phenotypic space
list_n = [2, 5, 10, 20, 30, 50]
results = {}
for n in list_n:
    initial_position = np.ones(n)*10/np.sqrt(n) 
    fgm = FisherGeometricModel(n, initial_position, population_size, alpha, Q, sigma_mut, duplication_rate, deletion_rate, mutation_rate)
    memory, fitness, effects = fgm.evolve(n_generations)  
    results[n] = fitness

plt.figure()
for n, fitness in results.items():
    plt.plot(fitness, label=f'n_traits = {n}')

plt.xlabel('Time')
plt.ylabel('Fitness')
plt.title('Evolution of Fitness Over Time with Different Numbers of Traits')
plt.legend()
plt.show()"""

# cost of complexity :
results = {}
for i in range(100):
    fgm = FisherGeometricModel(n_traits, initial_position, population_size, alpha, Q, sigma_mut, duplication_rate, deletion_rate, mutation_rate)
    memory, fitness, effects = fgm.evolve(n_generations)  
    results[fitness[-1]] = len(fgm.genes)

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
print(results)