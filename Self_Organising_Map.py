# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 15:16:36 2022

@author: Korisnik
"""




#%% Libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%% Loading data

data = pd.read_csv(r'C:\Users\Korisnik\OneDrive\Desktop\Neural Networks Chalmers-Gothenburg\HW3\Self Organising Map\iris-data.csv', header=None)
labels = pd.read_csv(r'C:\Users\Korisnik\OneDrive\Desktop\Neural Networks Chalmers-Gothenburg\HW3\Self Organising Map\iris-labels.csv', header=None)

#%% Normalizing data and converting to numpy array

data[0] /= data[0].max()
data[1] /= data[1].max()
data[2] /= data[2].max()
data[3] /= data[3].max()

data = data.to_numpy()
labels = labels.to_numpy()

#%% Parameters

lr = 0.1
lr_decay = 0.01

sigma = 10
sigma_decay = 0.05

epochs = 10

#%% Weight array

W_array = np.random.uniform(0, 1, (40, 40, 4))

#%% Plot of random organised map

# X, Y component in W_array of sample of nth class
X0 = []
Y0 = []
X1 = []
Y1 = []
X2 = []
Y2 = []

for sample in range(len(labels)):
    data_point = data[sample]
    
    best = float('inf')
    best_pos = [0, 0]

    for i in range(40):
        for j in range(40):
            dist = data_point - W_array[i][j]
            norm_dist = dist.T @ dist            
            if norm_dist < best:
                best = norm_dist
                best_pos = [i, j]

    if labels[sample, 0] == 0:
        X0.append(best_pos[0])
        Y0.append(best_pos[1])
    
    elif labels[sample, 0] == 1:
        X1.append(best_pos[0])
        Y1.append(best_pos[1])

    else:
        X2.append(best_pos[0])
        Y2.append(best_pos[1])
        
        
plt.scatter(X0, Y0, label = 'Label 0')
plt.scatter(X1, Y1, label = 'Label 1')
plt.scatter(X2, Y2, label = 'Label 2')
plt.legend()
plt.show()


#%% Neighbourhood function

def h(r, r0):
    global sigma
    
    distance = (r[0]-r0[0])**2 + (r[1]-r0[1])**2
    return( np.exp(-distance/(2*sigma)) )

#%% Learning phase

W_dist = np.empty( (40, 40, 4) )

for epoch in range(epochs):
    
    for data_point in data:
        
        best = float('inf')
        best_pos = [0, 0]
    
        # Finding the best weight vector
        for i in range(40):
            for j in range(40):
                dist = data_point - W_array[i][j]
                W_dist[i][j] = dist
                norm_dist = dist.T @ dist
            
                if norm_dist < best:
                    best = norm_dist
                    best_pos = [i, j]
        
        # Learning rule for weight vectors
        for i in range(40):
            for j in range(40):
                W_array[i, j] += lr * h([i, j], best_pos) * W_dist[i, j]
                
    lr *= np.exp( lr_decay*(epoch+1) )
    sigma *= np.exp( sigma_decay*(epoch+1) )
    
    
#%% Plot of self organised map

# X, Y component in W_array of sample of nth class
X0 = []
Y0 = []
X1 = []
Y1 = []
X2 = []
Y2 = []

for sample in range(len(labels)):
    data_point = data[sample]
    
    best = float('inf')
    best_pos = [0, 0]

    for i in range(40):
        for j in range(40):
            dist = data_point - W_array[i][j]
            norm_dist = dist.T @ dist            
            if norm_dist < best:
                best = norm_dist
                best_pos = [i, j]

    if labels[sample, 0] == 0:
        X0.append(best_pos[0])
        Y0.append(best_pos[1])
    
    elif labels[sample, 0] == 1:
        X1.append(best_pos[0])
        Y1.append(best_pos[1])

    else:
        X2.append(best_pos[0])
        Y2.append(best_pos[1])
        
        
plt.scatter(X0, Y0, label = 'Label 0')
plt.scatter(X1, Y1, label = 'Label 1')
plt.scatter(X2, Y2, label = 'Label 2')
plt.legend()
plt.show()
    






















