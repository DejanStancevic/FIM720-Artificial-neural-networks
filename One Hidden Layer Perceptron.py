# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 14:12:02 2022

@author: Korisnik
"""




#%% Importing Libraries and Data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

training_path = r'C:\Users\Korisnik\OneDrive\Desktop\Neural Networks Chalmers-Gothenburg\HW2\One_Layer_Perceptron\training_set.csv'
validation_path = r'C:\Users\Korisnik\OneDrive\Desktop\Neural Networks Chalmers-Gothenburg\HW2\One_Layer_Perceptron\validation_set.csv'

inputs = ['x_1', 'x_2']
target = ['target']
training_data = pd.read_csv(training_path, names = inputs+target)
validation_data = pd.read_csv(validation_path, names = inputs+target)


#%% Getting familiar with the data

print(type(training_data))

print(training_data[inputs])

#%% Normalizing the data

means = training_data[inputs].mean()
stds = training_data[inputs].std()

training_data[inputs] = (training_data[inputs] - means)/stds
validation_data[inputs] = (validation_data[inputs] - means)/stds

#%%

plt.scatter(training_data[training_data['target'] == 1]['x_1'], training_data[training_data['target'] == 1]['x_2'])
plt.scatter(training_data[training_data['target'] == -1]['x_1'], training_data[training_data['target'] == -1]['x_2'])
plt.show()

#%%

plt.scatter(validation_data[validation_data['target'] == 1]['x_1'], validation_data[validation_data['target'] == 1]['x_2'])
plt.scatter(validation_data[validation_data['target'] == -1]['x_1'], validation_data[validation_data['target'] == -1]['x_2'])
plt.show()

#%% Creating a Model

class Net:
    
    def __init__(self, M1):
        self.M1 = M1
        self.weights1 = np.random.normal(0, 1, (self.M1, 2))
        self.weights2 = np.random.normal(0, 1, (1, self.M1))
        self.bias1 = np.zeros((self.M1, 1))
        self.bias2 = np.zeros((1, 1))
        
    def g(b):
        return np.tanh(b)
    def dg(b):
        return (1 - (np.tanh(b))**2)
    
    def forward(self, inputs):
        self.localfield1 = self.weights1 @ inputs - self.bias1
        self.localfield2 = self.weights2 @  Net.g(self.localfield1) - self.bias2
    
    def Loss(outputs, targets):
        return (1/2) * (outputs-targets)**2
        
    def backward(self, outputs, targets):
        self.error2 = (outputs - targets) * Net.dg(self.localfield2)
        self.error1 = (self.error2.T @ self.weights2).T * Net.dg(self.localfield1)
    
    def update(self, inputs, targets, lr = 10**(-3), batch_size = 1):
        delta_weights2 = 0
        delta_bias2 = 0
        delta_weights1 = 0
        delta_bias1 = 0
        
        for i in range(batch_size):
            self.forward(inputs[i])
            self.backward(Net.g(self.localfield2), targets[i])
            
            delta_weights2 -= self.error2 @ Net.g(self.localfield1).T
            delta_bias2 += self.error2
        
            delta_weights1 -= self.error1 @ inputs[i].T
            delta_bias1 += self.error1
            
        
        self.weights2 += lr * delta_weights2
        self.bias2 += lr * delta_bias2
        
        self.weights1 += lr * delta_weights1
        self.bias1 += lr * delta_bias1
        
    
    def __call__(self, inputs):
        self.forward(inputs)
        return Net.g(self.localfield2)
    
#%% Initializing the Model

model = Net(16)
    
#%% Training the Model

epochs = 64
batch_size = 32

for epoch in range(epochs):
    print(epoch)
    
    for j in range( round(len(training_data)/batch_size) ):
        
        batch = np.random.randint(0, len(training_data), batch_size)

        inputs = np.array( [ [[training_data.iloc[i]['x_1']],
                              [training_data.iloc[i]['x_2']]] for i in batch] )
        targets = np.array([ training_data.iloc[i]['target'] for i in batch ])
        
        model.update(inputs, targets, lr = 0.03, batch_size = batch_size)
        

#%% Validating the Model

Total_Loss = 0
for i in range(len(validation_data)):
    data = validation_data.iloc[i]
    inputs = np.array( [[data['x_1']],
                       [data['x_2']]] )
    targets = data['target']
    
    outputs = model(inputs)

    
    if np.sign(outputs)[0][0] == targets:
        pass
    else:
        Total_Loss += 1
    
print(Total_Loss)

#%%

print( len(validation_data[validation_data['target'] == 1]) )

print( len(training_data[training_data['target'] == 1]) )
print(len(training_data))

#%%


np.savetxt(r"C:\Users\Korisnik\OneDrive\Desktop\Neural Networks Chalmers-Gothenburg\HW2\One_Layer_Perceptron\w2.csv", model.weights2, delimiter=",")
np.savetxt(r"C:\Users\Korisnik\OneDrive\Desktop\Neural Networks Chalmers-Gothenburg\HW2\One_Layer_Perceptron\w1.csv", model.weights1, delimiter=",")
np.savetxt(r"C:\Users\Korisnik\OneDrive\Desktop\Neural Networks Chalmers-Gothenburg\HW2\One_Layer_Perceptron\t2.csv", model.bias2, delimiter=",")
np.savetxt(r"C:\Users\Korisnik\OneDrive\Desktop\Neural Networks Chalmers-Gothenburg\HW2\One_Layer_Perceptron\t1.csv", model.bias1, delimiter=",")




        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
