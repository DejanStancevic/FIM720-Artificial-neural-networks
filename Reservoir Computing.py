# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 11:15:10 2022

@author: Korisnik
"""




#%% Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

#%% Uploading Data

training_path = r"C:\Users\Korisnik\OneDrive\Desktop\Neural Networks Chalmers-Gothenburg\HW3\Reservoir\training-set.csv"
test_path = r"C:\Users\Korisnik\OneDrive\Desktop\Neural Networks Chalmers-Gothenburg\HW3\Reservoir\test-set-5.csv"

T_training = [i for i in range(1, 19900+1)]
T_test = [i for i in range(1, 100+1)]

training = pd.read_csv(training_path, names = T_training)
test = pd.read_csv(test_path, names = T_test)

#%% Training outputs matrix

OUTPUTS_training = np.array([[training[2][0]],
                            [training[2][1]],
                            [training[2][2]]])

for i in range(2, training.shape[1]):
    
    OUTPUTS_training = np.append(OUTPUTS_training, [[training[i+1][0]],
                                                  [training[i+1][1]],
                                                  [training[i+1][2]]], axis = 1)



#%% Creating a Reservoir class

class Reservoir():
    
    def __init__(self, res_size = 500, ins = 3, outs = 3):
        self.outs = None
        self.reservoir = np.zeros((res_size, 1))
        
        self.weights_ins = np.random.normal(0, np.sqrt(0.002), (res_size, ins))
        self.weights_res = np.random.normal(0, np.sqrt(2/500), (res_size, res_size))
        self.weights_outs = None
        
    def update(self, ins):
        self.reservoir = np.tanh(self.weights_res @ self.reservoir + self.weights_ins @ ins)
        
    def get_outs(self):
        self.outs = self.weights_outs @ self.reservoir
        
    def train(self, res, outs, k = 0.01):
        rrt = res @ res.T
        I = np.identity(len(rrt))
        self.weights_outs = (outs @ res.T) @ np.linalg.inv( rrt + k * I )
        
        
    
#%% Updating through the training data

my_res = Reservoir()
my_res.update(np.array([ [training[1][0]],
                              [training[1][1]],
                              [training[1][2]] ]))


RESERVOIR_training = np.zeros((500, 19899))

RESERVOIR_training[:, [0]] = my_res.reservoir

for i in range(2, OUTPUTS_training.shape[1]+1):
    my_res.update(np.array([ [training[i][0]],
                              [training[i][1]],
                              [training[i][2]] ]))
    RESERVOIR_training[:, [i-1]] = my_res.reservoir


print(RESERVOIR_training.shape) # res_size X Time - 1


#%% Learning output weights

my_res.train(RESERVOIR_training, OUTPUTS_training)

#%% Running through the known test data

my_res.reservoir = np.zeros( (len(RESERVOIR_training), 1) )

for i in range(1, test.shape[1]+1):
    my_res.update(np.array([ [test[i][0]],
                              [test[i][1]],
                              [test[i][2]] ]))
    
#%% Predicting the future for T = 500 steps

FUTURE = []

T = 500

for t in range(T):
    my_res.get_outs()
    FUTURE.append(my_res.outs)
    my_res.update(my_res.outs)
    
    
FUTURE = np.array(FUTURE)

#%% Saving the predictions

np.savetxt(r"C:\Users\Korisnik\OneDrive\Desktop\Neural Networks Chalmers-Gothenburg\HW3\Reservoir\prediction.csv", FUTURE[:, 1], delimiter = ",")

        
#%% Plotting the predictions


ax = plt.axes(projection='3d')

FUTUREX = np.array([x[0] for x in FUTURE])
FUTUREY = np.array([x[1] for x in FUTURE])
FUTUREZ = np.array([x[2] for x in FUTURE])

print(FUTUREX[::4])

ax.scatter3D(FUTUREX, FUTUREY, FUTUREZ)
plt.show()
        
        
        
        
        
        
        
        
        
        
        
        