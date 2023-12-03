# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 23:36:14 2022

@author: Korisnik
"""




#%% HOPFIELD MODEL AND LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import random as r

class Hopfield:
    
    def __init__(self, height_x_width = tuple, noise = 0, diagonal = 0):
        self.noise = noise
        self.diagonal = diagonal
        self.height = height_x_width[0]
        self.width = height_x_width[1]
        self.neurons = np.zeros((self.height * self.width, 1))
        self.N = self.height * self.width # Number of neurons
        self.weights = np.zeros((self.N, self.N))
        
    def mat_to_vector(matrix):
        matrix = np.array(matrix)
        matrix.resize((matrix.size, 1))
        return(matrix)
    
    def sign(number):
        if number <0:
            return(-1)
        else:
            return(1)
        
    def store_pattern(self, pattern):
        assert len(pattern) == self.N
        self.weights += pattern @ pattern.T / self.N
        
        if not self.diagonal:
            self.weights -= np.identity(self.N)/self.N
            
    def initial_state(self, state):
        self.neurons = state.copy()
            
    def neuron_update(self, neuron_position):
        if self.noise == 0:
            self.neurons[neuron_position][0] = Hopfield.sign((self.weights @ self.neurons)[neuron_position][0])
        else:
            pass
            # UNDER CONSTRUCTION
            
#%% HOMEWORK: DIGITS
    
PATTERNS = [ 
            [ [ -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],[ -1, -1, 1, 1, 1, 1, 1, 1, -1, -1],[ -1, 1, 1, 1, -1, -1, 1, 1, 1, -1],[ -1, 1, 1, 1, -1, -1, 1, 1, 1, -1],[ -1, 1, 1, 1, -1, -1, 1, 1, 1, -1],[ -1, 1, 1, 1, -1, -1, 1, 1, 1, -1],[ -1, 1, 1, 1, -1, -1, 1, 1, 1, -1],[ -1, 1, 1, 1, -1, -1, 1, 1, 1, -1],[ -1, 1, 1, 1, -1, -1, 1, 1, 1, -1],[ -1, 1, 1, 1, -1, -1, 1, 1, 1, -1],[ -1, 1, 1, 1, -1, -1, 1, 1, 1, -1],[ -1, 1, 1, 1, -1, -1, 1, 1, 1, -1],[ -1, -1, 1, 1, 1, 1, 1, 1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],[ -1, -1, -1, -1, -1, -1, -1, -1, -1, -1] ],
            [ [ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1] ],
            [ [ 1, 1, 1, 1, 1, 1, 1, 1, -1, -1],[ 1, 1, 1, 1, 1, 1, 1, 1, -1, -1],[ -1, -1, -1, -1, -1, 1, 1, 1, -1, -1],[ -1, -1, -1, -1, -1, 1, 1, 1, -1, -1],[ -1, -1, -1, -1, -1, 1, 1, 1, -1, -1],[ -1, -1, -1, -1, -1, 1, 1, 1, -1, -1],[ -1, -1, -1, -1, -1, 1, 1, 1, -1, -1],[ 1, 1, 1, 1, 1, 1, 1, 1, -1, -1],[ 1, 1, 1, 1, 1, 1, 1, 1, -1, -1],[ 1, 1, 1, -1, -1, -1, -1, -1, -1, -1],[ 1, 1, 1, -1, -1, -1, -1, -1, -1, -1],[ 1, 1, 1, -1, -1, -1, -1, -1, -1, -1],[ 1, 1, 1, -1, -1, -1, -1, -1, -1, -1],[ 1, 1, 1, -1, -1, -1, -1, -1, -1, -1],[ 1, 1, 1, 1, 1, 1, 1, 1, -1, -1],[ 1, 1, 1, 1, 1, 1, 1, 1, -1, -1] ],
            [ [ -1, -1, 1, 1, 1, 1, 1, 1, -1, -1],[ -1, -1, 1, 1, 1, 1, 1, 1, 1, -1],[ -1, -1, -1, -1, -1, -1, 1, 1, 1, -1],[ -1, -1, -1, -1, -1, -1, 1, 1, 1, -1],[ -1, -1, -1, -1, -1, -1, 1, 1, 1, -1],[ -1, -1, -1, -1, -1, -1, 1, 1, 1, -1],[ -1, -1, -1, -1, -1, -1, 1, 1, 1, -1],[ -1, -1, 1, 1, 1, 1, 1, 1, -1, -1],[ -1, -1, 1, 1, 1, 1, 1, 1, -1, -1],[ -1, -1, -1, -1, -1, -1, 1, 1, 1, -1],[ -1, -1, -1, -1, -1, -1, 1, 1, 1, -1],[ -1, -1, -1, -1, -1, -1, 1, 1, 1, -1],[ -1, -1, -1, -1, -1, -1, 1, 1, 1, -1],[ -1, -1, -1, -1, -1, -1, 1, 1, 1, -1],[ -1, -1, 1, 1, 1, 1, 1, 1, 1, -1],[ -1, -1, 1, 1, 1, 1, 1, 1, -1, -1] ],
            [ [ -1, 1, 1, -1, -1, -1, -1, 1, 1, -1],[ -1, 1, 1, -1, -1, -1, -1, 1, 1, -1],[ -1, 1, 1, -1, -1, -1, -1, 1, 1, -1],[ -1, 1, 1, -1, -1, -1, -1, 1, 1, -1],[ -1, 1, 1, -1, -1, -1, -1, 1, 1, -1],[ -1, 1, 1, -1, -1, -1, -1, 1, 1, -1],[ -1, 1, 1, -1, -1, -1, -1, 1, 1, -1],[ -1, 1, 1, 1, 1, 1, 1, 1, 1, -1],[ -1, 1, 1, 1, 1, 1, 1, 1, 1, -1],[ -1, -1, -1, -1, -1, -1, -1, 1, 1, -1],[ -1, -1, -1, -1, -1, -1, -1, 1, 1, -1],[ -1, -1, -1, -1, -1, -1, -1, 1, 1, -1],[ -1, -1, -1, -1, -1, -1, -1, 1, 1, -1],[ -1, -1, -1, -1, -1, -1, -1, 1, 1, -1],[ -1, -1, -1, -1, -1, -1, -1, 1, 1, -1],[ -1, -1, -1, -1, -1, -1, -1, 1, 1, -1] ]
    ]        
            
DIGITS = Hopfield((len(PATTERNS[0]), len(PATTERNS[0][0])))


for i in range(len(PATTERNS)):
    PATTERNS[i] = Hopfield.mat_to_vector(PATTERNS[i])
    

for pattern in PATTERNS:
    DIGITS.store_pattern(pattern)

    
state = [[1, 1, -1, -1, -1, -1, -1, -1, 1, 1], [-1, -1, 1, 1, 1, 1, 1, 1, 1, -1], [-1, -1, -1, -1, -1, -1, 1, 1, 1, -1], [-1, -1, -1, -1, -1, -1, 1, 1, 1, -1], [-1, -1, -1, -1, -1, -1, 1, 1, 1, -1], [-1, -1, -1, -1, -1, -1, 1, 1, 1, -1], [-1, -1, -1, -1, -1, -1, 1, 1, 1, -1], [-1, -1, 1, 1, 1, 1, 1, 1, -1, -1], [-1, -1, 1, 1, 1, 1, 1, 1, -1, -1], [-1, -1, -1, -1, -1, -1, 1, 1, 1, -1], [-1, -1, -1, -1, -1, -1, 1, 1, 1, -1], [-1, -1, -1, -1, -1, -1, 1, 1, 1, -1], [-1, -1, -1, -1, -1, -1, 1, 1, 1, -1], [-1, -1, -1, -1, -1, -1, 1, 1, 1, -1], [-1, -1, 1, 1, 1, 1, 1, 1, 1, -1], [-1, -1, 1, 1, 1, 1, 1, 1, -1, -1]]
state = Hopfield.mat_to_vector(state)
DIGITS.initial_state(state)



step = 0
last_state = np.zeros((DIGITS.N, 1))

plt.imshow(state.reshape((16, 10)), cmap='hot')
plt.show()

while np.any(last_state != state) and step < 11:
    last_state = state.copy()
    for i in range(DIGITS.N):
        DIGITS.neuron_update(i)
    state = DIGITS.neurons
    step += 1
    
print(step)

print( [list(row) for row in state.reshape(16, 10)] )



#%% IMAGE OF A NEURONS' STATE
plt.imshow(state.reshape((16, 10)), cmap='hot')
plt.show()


#%% IMAGE OF A PATTERN
plt.imshow(PATTERNS[3].reshape((16, 10)), cmap='hot')
plt.show()

#%% HOMEWORK: ONE STEP PROBABILITY FUNCTION

p = [12,24,48,70,100,120]
P = [] # Probabilities

neuron_values = [-1, 1]

trails = 10**5
N = 120

for n in p:
    
    change = 0
    for trail in range(trails):
        
        PATTERNS = []
        ONE_STEP = Hopfield((N, 1), diagonal = 1)
        for i in range(n):
            pattern = np.random.choice(neuron_values, size = (N, 1))
            PATTERNS.append(pattern)
            ONE_STEP.store_pattern(pattern)

        neuron = r.randint(0, N-1)
        state = r.choice(PATTERNS)
        
        if Hopfield.sign((ONE_STEP.weights @ state)[neuron][0]) != state[neuron][0]:
            change += 1
    
    P.append(round(change/trails, 4))
    print(P)
    
print(P)
        
    



            
            
    
    
        
        
            



