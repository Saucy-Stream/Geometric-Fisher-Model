import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.special import gamma, gammaincc

class FisherGeometricModel() :
    def __init__(self, n, initial_position, d) :
        self.dimension = n
        self.optimum = np.zeros(n)
        self.position = initial_position
        self.selection_matrix(d)

    def selection_matrix(self, d):
        v = np.random.multivariate_normal(np.zeros(self.dimension), d*np.eye(self.dimension), self.dimension)
        self.S = 1/self.dimension * sum(np.outer(vi, vi) for vi in v)
        # print(self.S)
        
    def mutation_matrix(self, ml, cl):
        # t1 = time.time()
        v = np.random.multivariate_normal(np.zeros(self.dimension), cl*np.eye(self.dimension), ml)
        Ml = 1/ml * sum(np.outer(vi, vi) for vi in v)
        # print(Ml)
        # print(time.time() - t1)
        return Ml

    def mutation(self, Ml):
        dz = np.random.multivariate_normal(np.zeros(self.dimension), Ml) 
        new_position = self.position + dz
        return dz, new_position

    def fitness_function(self, position):
        w = np.exp(-1/2 * (position.T @ self.S @ position))
        return w

    def fitness_effect(self, dz):
        return -1/2*dz.T@self.S@dz - self.position.T@self.S@dz
        # return new_fitness/initial_fitness - 1 # Martin 2006 (for s small)
        # if > 0, the mutation as improve the fitness, it is beneficial
    
    def fitness_effect_v2(self, initial_fitness, new_fitness) :
        return np.log(new_fitness) - np.log(initial_fitness)

    def fixation_probability(self, s) :
        # p = 2*s # Haldane 1927 (only viable for very little s)
        p = (1 - np.exp(-2*s)) / (1 - np.exp(-2*self.N*s)) # Barrett 2006
        return p
        
    def fixation_probability_v2(self, ml, Ml) :
        s0 = self.position.T @ self.S @ self.position
        smax = 1/2 * s0
        pfl = 4*smax/(2+ml)

        trace = np.trace(self.S@Ml)
        El = -1/2 * trace
        epsilon = smax/np.abs(El)
        ql = (self.position.T @ (self.S)**2 @ self.position) / s0 * (trace/np.trace((self.S@Ml)**2))
        # print(ql) # ql très éloigné de 1 ici ??
        # ql = 1 d'après l'article (effet faible devant smax, vérifié par simulation)
        ne = self.dimension/(2 + self.dimension/ml)
        a = -2*El/ne
        alpha = a * (1 + 2*ql*epsilon)/(1 + epsilon)
        beta = ne/2 * (1 + epsilon)**2/(1 + 2*ql*epsilon)

        pbl = 1 - gammaincc(beta, smax/alpha) / gamma(beta)

        Pl = pbl*pfl
        return Pl

    def evolve(self, time_step, sigma_mut) :
        memory = [self.position]
        fitness = []
        effects = []

        for i in range(time_step):
            ml = np.random.randint(1, 51)
            cl = sigma_mut
            Ml = self.mutation_matrix(ml, cl)
            initial_fitness = self.fitness_function(self.position)
            # print(initial_fitness)
            dz, new_position = self.mutation(Ml)
            # print(new_position)
            new_fitness  = self.fitness_function(new_position)
            # print(new_fitness)
            s = self.fitness_effect_v2(initial_fitness, new_fitness)
            if s > 0 :
                # print(np.linalg.norm(self.position))
                # print(np.linalg.norm(new_position)) 
                # pourquoi le fitness peut être meilleur même quand on s'éloigne de l'origine ??
                # selon les coefficients de S : certains traits plus importants que d'autres 
                # --> si la position change sur ces traits donne un meilleur fitness même si on s'éloigne de l'origine ? 
                # Les mutations qui sont bénéfiques pour les traits les plus "importants" sont conservées
                pf = self.fixation_probability_v2(ml, Ml)
                # print(pf) # les probas sont très élevées comparer au modèle de Fisher standard, ...

                if np.random.rand() < pf :
                    self.position = new_position
                    memory.append(new_position)
                    fitness.append(new_fitness)
            else :
                fitness.append(initial_fitness)

            effects.append(s)
                
        return memory, fitness, effects

    def ploting_results(self, fitness, effects):
        plt.figure()

        plt.subplot(1, 2, 1)
        plt.plot(fitness)
        plt.xlabel('Time')
        plt.ylabel('Fitness')
        plt.title('Evolution of Fitness Over Time')

        plt.subplot(1, 2, 2)
        plt.plot(effects, '.', markersize=3)
        plt.hlines(0, 0, 10000, "k", "--")
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
initial_position = np.ones(n_traits)*10/np.sqrt(50) 
n_generations = 10000  # Number of generations to simulate
r = 0.5 
sigma_mut = r/np.sqrt(n_traits) # Standard deviation of the mutation effect size # Tenaillon 2014
# here sigma is the same on every dimension
# population_size = 1  # Effective population size N
d = 1


# Simulation
fgm = FisherGeometricModel(n_traits, initial_position, d)
memory, fitness, effects = fgm.evolve(n_generations, sigma_mut)
fgm.ploting_results(fitness, effects)
fgm.ploting_path(memory)

