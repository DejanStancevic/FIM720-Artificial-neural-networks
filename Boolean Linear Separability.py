# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 13:38:44 2022

@author: Korisnik
"""




#%% Perceptron and Libraries
import numpy as np
import itertools as it
import random

class Perceptron:
    
    def __init__(self, num_inpts, lr = 0.05):
        self.weights = np.random.normal(0, 1/num_inpts, (num_inpts, 1))
        self.bias = 0
        self.lr = lr # Learning Rate
        
    def output(self, inputs):
        local_field = self.weights.T @ inputs - self.bias
        
        if local_field < 0:
            return(-1)
        else:
            return(1)
    
    def update(self, inputs, target):
        difference = target - self.output(inputs)
        self.weights += self.lr * difference * inputs
        self.bias -= self.lr * difference
        
BOOL = [0, 1]
epochs = 20

#%% n = 2, ls = 14
n = 2
total = 2**(2**n) # Total number of boolean function in n dimensions

INPUTS = []


for i in range(n+1):
    for combination in it.combinations(range(n), i):
        inputs = np.zeros((n, 1))
        for position in combination:
            inputs[position][0] = 1
        INPUTS.append(inputs)

TARGETS = []

for i in range(2**n+1):
    for combination in it.combinations(range(2**n), i):
        targets = [-1]*2**n
        for position in combination:
            targets[position] = 1
        TARGETS.append(targets)
        
        
nls = 0 # Number of non-linearly separable boolean functions

for target in TARGETS: # Each target is a separate boolean function
    
    perceptron = Perceptron(n)
    
    for epoch in range(epochs):
        for i in range(len(INPUTS)):
            perceptron.update(INPUTS[i], target[i])
    
    for i in range(len(INPUTS)):
        if perceptron.output(INPUTS[i]) != target[i]:
            nls += 1
            break
        
print(total - nls)




#%% n = 3, ls = 104
n = 3
total = 2**(2**n) # Total number of boolean function in n dimensions

INPUTS = []


for i in range(n+1):
    for combination in it.combinations(range(n), i):
        inputs = np.zeros((n, 1))
        for position in combination:
            inputs[position][0] = 1
        INPUTS.append(inputs)

TARGETS = []

for i in range(2**n+1):
    for combination in it.combinations(range(2**n), i):
        targets = [-1]*2**n
        for position in combination:
            targets[position] = 1
        TARGETS.append(targets)
        
        
nls = 0 # Number of non-linearly separable boolean functions

for target in TARGETS: # Each target is a separate boolean function
    
    perceptron = Perceptron(n)
    
    for epoch in range(epochs):
        for i in range(len(INPUTS)):
            perceptron.update(INPUTS[i], target[i])
    
    for i in range(len(INPUTS)):
        if perceptron.output(INPUTS[i]) != target[i]:
            nls += 1
            break
        
print(total - nls)
        

#%% n = 4, ls = 1901
n = 4
num_bool = 10**4 # Number of boolean functions sampled
total = 2**(2**n) # Total number of boolean function in n dimensions

INPUTS = []


for i in range(n+1):
    for combination in it.combinations(range(n), i):
        inputs = np.zeros((n, 1))
        for position in combination:
            inputs[position][0] = 1
        INPUTS.append(inputs)

TARGETS = []
        
        
nls = 0 # Number of non-linearly separable boolean functions

for bool_func in range(num_bool): # Each target is a separate boolean function
    
    perceptron = Perceptron(n)
    
    while True:
        target = [random.choice([-1, 1]) for i in range(2**n)]
        if not (target in TARGETS):
            TARGETS.append(target)
            break
    
    for epoch in range(epochs):
        for i in range(len(INPUTS)):
            perceptron.update(INPUTS[i], target[i])
    
    for i in range(len(INPUTS)):
        if perceptron.output(INPUTS[i]) != target[i]:
            nls += 1
            break
        
ratio = total/num_bool
print(round(total - nls * ratio)) # Approximate number of non-linearlly separable boolean functions




#%% n = 5, ls = 0, 429 497
n = 5
num_bool = 10**4 # Number of boolean functions sampled
total = 2**(2**n) # Total number of boolean function in n dimensions

INPUTS = []


for i in range(n+1):
    for combination in it.combinations(range(n), i):
        inputs = np.zeros((n, 1))
        for position in combination:
            inputs[position][0] = 1
        INPUTS.append(inputs)

TARGETS = []
        
        
nls = 0 # Number of non-linearly separable boolean functions

for bool_func in range(num_bool): # Each target is a separate boolean function
    
    perceptron = Perceptron(n)
    
    while True:
        target = [random.choice([-1, 1]) for i in range(2**n)]
        if not (target in TARGETS):
            TARGETS.append(target)
            break
    
    for epoch in range(epochs):
        for i in range(len(INPUTS)):
            perceptron.update(INPUTS[i], target[i])
    
    for i in range(len(INPUTS)):
        if perceptron.output(INPUTS[i]) != target[i]:
            nls += 1
            break
        
ratio = total/num_bool
print(round(total - nls * ratio)) # Approximate number of non-linearlly separable boolean functions
    
            
    



