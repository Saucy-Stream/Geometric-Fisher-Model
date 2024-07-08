import numpy as np
import random
import math

def fitness_function(alpha, Q, d):
    return np.exp(-alpha*d**Q)

def standardized_mutational_size(r, n, d):
    return r*np.sqrt(n)/(2*d)

def benefical_mutation_fraction(x):
    return 1/2*math.erfc(x/np.sqrt(2))

def distance_to_optimum(z):
    return np.sqrt(sum(z**2))

def simple_FGM(z, r, n, time_step, alpha, Q):
    for i in range(time_step):
        Dz = random.gauss(mu = -r**2/(2*z), sigma = np.sqrt(r**2/n)) # Orr 2006
        # print(Dz)
        if Dz > 0 : # mutation is benefical
            s = fitness_function(alpha, Q, z-Dz)/fitness_function(alpha, Q, z) - 1 # Guillaume Martin 2006
            # s = z*Dz # Orr 2000
            fixating_probability = 2*s # Orr 2000
            p = random.uniform(a = 0, b = 1)
            if p < fixating_probability : # mutation is fixed
                z = z - Dz
                print(z)


##### Parameters #####
n = 100
Q = 2
alpha = 0.5
r = 0.5
init = 10

simple_FGM(init, r, n, 100000, alpha, Q)
