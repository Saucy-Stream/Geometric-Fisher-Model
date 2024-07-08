import numpy as np
import matplotlib.pyplot as plt

class FisherGeometricModel() :
    def __init__(self, n, initial_position, size) :
        self.dimension = n
        self.optimum = np.zeros(n)
        self.position = initial_position
        self.N = size

    def mutation(self, sigma_mut):
        m = np.random.normal(0, sigma_mut, self.dimension) # Tenaillon 2014 / Blanchart 2014
        new_position = self.position + m
        return new_position

    def fitness_function(self, alpha, Q, position):
        d = np.linalg.norm(position)
        # print(d)
        w = np.exp(-alpha * d**Q)
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
        return p

    def evolve(self, time_step, sigma_mut, alpha, Q) :
        memory = [self.position]
        fitness = []
        effects = []

        for i in range(time_step):
            initial_fitness = self.fitness_function(alpha, Q, self.position)
            # print(initial_fitness)
            new_position = self.mutation(sigma_mut)
            # print(new_position)
            new_fitness  = self.fitness_function(alpha, Q, new_position)
            # print(new_fitness)
            s = self.fitness_effect(initial_fitness, new_fitness)
            
            pf = self.fixation_probability(s)
            # print(pf)

            if np.random.rand() < pf :
                self.position = new_position
                memory.append(new_position)
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
initial_position = np.ones(n_traits)*10/np.sqrt(n_traits) # Quand la position initiale est plus éloigné de l'origine, la pop à bcp moins de mal à s'améliorer (et les mutations sont plus grandes ?)
n_generations = 10**5  # Number of generations to simulate (pas vraiment, voir commentaire sur Nu)
r = 0.5 
sigma_mut = r/np.sqrt(n_traits) # Standard deviation of the mutation effect size # Tenaillon 2014
# here sigma is the same on every dimension
population_size = 10**4  # Effective population size N
alpha = 1/2
Q = 2
mutation_rate = 10**(-6) # rate of mutation mu
# La simulation actuelle à donc une echelle de temps en (Nu)**(-1) soit une mutation toute les 100 générations

# Simulation
fgm = FisherGeometricModel(n_traits, initial_position, population_size)
memory, fitness, effects = fgm.evolve(n_generations, sigma_mut, alpha, Q)
fgm.ploting_results(fitness, effects, n_generations)
fgm.ploting_path(memory)

# ajouter probabilité de fixation d'une mutation neutre : 1/2N ? (Kimura)
# proba de fixation d'une mutation quasi neutre (s<1/2N) (Ohta)

"""
plt.figure()
time = [t for t in range(n_generations)]
c = 1
list_n = [2, 5, 10, 20, 30, 50]
for n in list_n:
    initial_position = np.ones(n)*1/np.sqrt(n) 
    fgm = FisherGeometricModel(n, initial_position, population_size)
    memory, fitness, effects = fgm.evolve(n_generations, sigma_mut, alpha, Q)
    df = []
    for i in range(1, n_generations):
        # print(i)
        tmp = (fitness[i]-fitness[i-1])/(time[i]-time[i-1])
        df.append(tmp)
    
    plt.subplot(2,3,c)
    plt.plot(time[1:], df, ".")
    plt.xlabel('Time')
    plt.ylabel('dw/dt')
    plt.title(f'Rate of adaptation for complexity n = {n}')
    # pas fameux car le fitness évolue de façon discrète, pas continu...
    # Le cout de la complexité est quand même clairement visible (regarder directement l'évolution du fitness)
    c += 1
    c += 1
plt.show()
"""

list_n = [2, 5, 10, 20, 30, 50]
results = {}
for n in list_n:
    initial_position = np.ones(n)*1/np.sqrt(n) 
    fgm = FisherGeometricModel(n, initial_position, population_size)
    memory, fitness, effects = fgm.evolve(n_generations, sigma_mut, alpha, Q)  
    results[n] = fitness

plt.figure()
for n, fitness in results.items():
    plt.plot(fitness, label=f'n_traits = {n}')

plt.xlabel('Time')
plt.ylabel('Fitness')
plt.title('Evolution of Fitness Over Time with Different Numbers of Traits')
plt.legend()
plt.show()
# mieux, voir version 2 pour étude en ne fixant que des mutations bénéfiques
# ou bien simplement augmenter N : quand la taille de la population est grande, les mutations neutres et quasi neutres ne se fixent pas.