# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 00:01:36 2022

@author: Korisnik
"""




#%% Importing Libraries and Creating Data

import numpy as np
import random as r
import math as m
import matplotlib.pyplot as plt

XOR = [ [-1, -1, -1],
        [1, -1, 1],
        [-1, 1, 1],
        [1, 1, -1]]

XOR = [np.array([ [x[0]],
                 [x[1]],
                 [x[2]]] ) for x in XOR]

print(XOR[0].shape)

#%% RBM class

class RBM:
    
    def __init__(self, H, V = 3):
        self.H = H
        self.V = V
        self.hidden, self.visible = np.zeros((1, H)), np.zeros((V, 1))
        self.weights = np.random.normal(0, 1, (H, V))
        self.hbias = np.zeros((1, H))
        self.vbias = np.zeros((V, 1))
        
    def update_v(self):
        v_local_field = (self.hidden @ self.weights).T - self.vbias
        self.visible = np.array( [[+1] if r.random() < (1+ m.exp(-2*v_local_field[i][0]))**(-1) else [-1] for i in range(self.V)] )
 
    def update_h(self):
        h_local_field = (self.weights @ self.visible).T - self.hbias
        self.hidden = np.array([ [+1 if r.random() < (1+ m.exp(-2*h_local_field[0][i]))**(-1) else -1 for i in range(self.H)] ])


    def train(self, data, k, lr, N_max = 10**(3), p0=20):
        #assert p0 <= len(data), 'p0 needs to be less or equal to len(data)'
        
        for i in range(N_max):
            
            delta_weights = 0
            delta_hbias = 0
            delta_vbias = 0
            
            for p in range(p0):
                data_i = np.random.randint(0, len(data))
                v0 = data[data_i]
                self.visible = v0
                h_local_field0 = (self.weights @ self.visible).T - self.hbias
                self.hidden = np.array([ [+1 if r.random() < (1+ m.exp(-2*h_local_field0[0][i]))**(-1) else -1 for i in range(self.H)] ])
            
                for t in range(k):
                    self.update_v()                
                    self.update_h()
            
                tanh0 = np.tanh(h_local_field0)
                tanh = np.tanh( (self.weights @ self.visible).T - self.hbias )
            
                delta_weights += tanh0.T @ v0.T - tanh.T @ self.visible.T
                delta_hbias -= tanh0 - tanh
                delta_vbias -= v0 - self.visible
                
            self.weights += lr * delta_weights
            self.hbias += lr * delta_hbias
            self.vbias += lr * delta_vbias
            

#%% Initializing Probabilities for Data and Model

def bool_to_num(vec):
    result = 0
    for i in range(len(vec)):
        if vec[i] > 0:
            result += 2**i
    return(result)

P_data = [0]*8
for x in XOR:
    P_data[bool_to_num(x)] = 1/4
    
#%% Determining K-L Divergence

def KLD(P_M, P_D):
    result = 0
    for i in range(len(P_M)):
        try:
            result += P_D[i] * m.log(P_D[i]/P_M[i])
        except:
            if P_D[i] == 0:
                pass
            else:
                print('INFINITY')
                break
    return(result)


D_kl = []
M = [1, 2, 4, 8]

for i in M:
    
    rbm = RBM(i)
    rbm.train(XOR, 2*10**(2), 0.005, 10**(3))
    
    start = XOR[r.randint(0, len(XOR)-1)]
    
    iterations = 10**(5)

    P_model = [0]*8
    rbm.visible = start
    for j in range(iterations):
        rbm.update_h()
        rbm.update_v()
        P_model[bool_to_num(rbm.visible)] += 1
    P_model = [p/iterations for p in P_model]
    
    D_kl.append( KLD(P_model, P_data) )
    
#%% K-L Divergence Theory

def KLD_theory(M, V = 3):
    if M < 2**(V-1) - 1:
        floor = m.floor( m.log(M+1, 2) )
        return( m.log(2) * (V - (M+1)/(2**(floor)) - floor) )
    else:
        return(0)    
        
D_kl_theory = []

for i in M:
    D_kl_theory.append( KLD_theory(i) )


    

#%% Plotting D_kl Theory vs Model
    

plt.scatter(M, D_kl, label = 'Model')
plt.plot(M, D_kl_theory, label = 'Theory')
plt.xlabel('Number of hidden neurons, M')
plt.ylabel('K-L Divergence')
plt.legend()
plt.show()

    
    

    
            
            
            
        
    



