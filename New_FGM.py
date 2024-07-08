import numpy as np
import matplotlib.pyplot as plt

class FisherGeometricModel() :
    def __init__(self, n, initial_position, size, alpha, Q, proba_duplication, proba_deletion, proba_mutation) :
        self.dimension = n
        self.N = size
        self.optimum = np.zeros(n)
        self.alpha = alpha
        self.Q = Q

        self.init_pos = initial_position
        self.genes = [self.create_first_gene()]

        self.final_pos = initial_position
        for gene in self.genes :
            self.final_pos += gene

        self.pdupl = proba_duplication
        self.pdel = proba_deletion
        self.pmut = proba_mutation
        
    def create_first_gene(self):
        s = -1 
        while s < 0 : # we consider that the first gene must be at least neutral or beneficial so that the organism survive
            gene = np.random.normal(0, sigma_mut, self.dimension) # on fait comme si c'était une mutation du point de départ dans le modèle FGM standart
            new_pos = self.init_pos + gene
            init_fitness = self.fitness_function(self.init_pos)
            fitness = self.fitness_function(new_pos)
            s = self.fitness_effect(init_fitness, fitness)

        return gene

    def mutation_on_one_gene(self, sigma_mut):
        # print(len(self.genes))
        index = np.random.randint(0, len(self.genes))
        m = np.random.normal(0, sigma_mut, self.dimension) # Tenaillon 2014 / Blanchart 2014
        new_gene = self.genes[index] + m

        new_final_pos = self.final_pos - self.genes[index] + new_gene
        return new_final_pos, new_gene, index
    
    def mutation_on_every_genes(self, sigma_mut):
        n = len(self.genes)
        m = np.random.normal(0, np.sqrt(n)*sigma_mut, size=(n, self.dimension))
        new_genes = [self.genes[i] + m[i] for i in range(n)]

        new_final_pos = self.init_pos
        for gene in new_genes :
            self.final_pos += gene
        return new_final_pos, new_genes
    
    def duplication(self):
        index = np.random.randint(0, len(self.genes))
        new_final_pos = self.final_pos + self.genes[index]
        return new_final_pos, index
    
    def deletion(self):
        index = np.random.randint(0, len(self.genes))
        new_final_pos = self.final_pos - self.genes[index]
        return new_final_pos, index

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

    def evolve(self, time_step, sigma_mut) :
        memory = [self.init_pos, self.final_pos]
        fitness = []
        effects = []
        methods = []

        for i in range(time_step):
            initial_fitness = self.fitness_function(self.final_pos)
            # print(initial_fitness)

            list_genes = self.genes.copy()
            p = np.random.rand()
            if p < self.pdel :
                new_final_pos, index = self.deletion()
                list_genes.pop(index)
                method = "deletion"
            elif p < self.pdupl : 
                new_final_pos, index = self.duplication()
                list_genes.append(self.genes[index])
                method = "duplication"
            else : 
                # If mutation on one gene : 
                new_final_pos, new_gene, index = self.mutation_on_one_gene(sigma_mut)
                list_genes[index] = new_gene

                # If mutation on every genes : 
                # new_final_pos, new_genes = self.mutation_on_every_genes(sigma_mut)
                # list_genes = new_genes

                method = "mutation"
            # print(new_final_pos)

            new_fitness  = self.fitness_function(new_final_pos)
            # print(new_fitness)
            s = self.fitness_effect(initial_fitness, new_fitness)
            pf = self.fixation_probability(s)
            # print(pf)

            if np.random.rand() < pf :
                self.genes = list_genes
                self.final_pos = new_final_pos
                memory.append(new_final_pos)
                methods.append(method)
                fitness.append(new_fitness)

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

# Parameters
n_traits = 50  # Number of traits in the phenotype space n
initial_position = np.ones(n_traits)*10/np.sqrt(n_traits) # Quand la position initiale est plus éloigné de l'origine, la pop à bcp moins de mal à s'améliorer (et les mutations sont plus grandes ?)
# first_gene (defini dans le init)
n_generations = 10**5  # Number of generations to simulate (pas vraiment, voir commentaire sur Nu)
r = 0.5 
sigma_mut = r/np.sqrt(n_traits) # Standard deviation of the mutation effect size # Tenaillon 2014
# here sigma is the same on every dimension
population_size = 10**4  # Effective population size N
alpha = 1/2
Q = 2
mutation_rate = 10**(-6) # rate of mutation mu
# La simulation actuelle à donc une echelle de temps en (Nu)**(-1) soit une mutation toute les 100 générations
duplication = 0.2 # %
duplication_rate = 10**(-7) # /gene/generation
deletion =  0.1 # %
mutation = 0.7 # % (if mutation at 100 and the other at 0, same as standart FGM but with genes instead of mutations)

# Simulation
fgm = FisherGeometricModel(n_traits, initial_position, population_size, alpha, Q, duplication, deletion, mutation)
memory, fitness, effects, methods = fgm.evolve(n_generations, sigma_mut)
fgm.ploting_results(fitness, effects, n_generations)
fgm.ploting_path(memory)

print(methods)

# complexity of the phenotypic space
list_n = [2, 5, 10, 20, 30, 50]
results = {}
for n in list_n:
    initial_position = np.ones(n)*10/np.sqrt(n) 
    fgm = FisherGeometricModel(n, initial_position, population_size, alpha, Q, duplication, deletion, mutation)
    memory, fitness, effects, methods = fgm.evolve(n_generations, sigma_mut)  
    results[n] = fitness

plt.figure()
for n, fitness in results.items():
    plt.plot(fitness, label=f'n_traits = {n}')

plt.xlabel('Time')
plt.ylabel('Fitness')
plt.title('Evolution of Fitness Over Time with Different Numbers of Traits')
plt.legend()
plt.show()