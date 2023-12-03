# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 16:05:24 2022

@author: Korisnik
"""




#%% Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F

#%% Loading MNIST data

import torchvision.datasets as datasets

MNIST = datasets.MNIST(root = r'C:\Users\Korisnik\OneDrive\Desktop\Neural Networks Chalmers-Gothenburg\HW2\MNIST', train=True, download=True, transform=None)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

#%% Getting familiar with the data

print(type(MNIST))
print(type(MNIST[0]))
print(type(MNIST[0][0]))
print(type(MNIST[0][1]))

display(MNIST[0][0])

#%% Creating tensors from PIL Images

data = []

for number in MNIST:
    tensor_image = (torch.tensor( np.array(number[0]) )[None, :]/256).float()
    
    data.append((tensor_image, number[1]))
    
#%% 

print(type(data[0]))
print(type(data[0][0]))
#%% MODEL

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        k = 5 # Kernel size
        
        self.conv1 = nn.Conv2d(1, 10, k)
        self.fc_layer1 = nn.Linear(24*24*10, 1000)
        self.fc_layer2 = nn.Linear(1000, 10)
        
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc_layer1(x))
        x = F.log_softmax(self.fc_layer2(x), dim = -1)
        
        return(x)
    
model = Model()

model.to(device)

#%% Training and Validation datasets

separation = 4*10**4

training = data[:separation]

validation = data[separation::]


#%% HYPERPARAMETERS

batch_size = 64
lr = 10**(-3) # Learning rate during the first two epochs. It gives one percent better result.

trainloader = torch.utils.data.DataLoader(training, batch_size = batch_size, shuffle = True)

#%% Optimizer

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

#%% Training

LOSS = []
my_loss = nn.NLLLoss()

epochs = 10

for epoch in range(epochs):
    print(epoch)
    
    if epoch < 6:
        pass
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=10**(-5))
        
    
    for i, (inputs, targets) in enumerate(trainloader):
        
        
        
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # set to zero the parameter gradients
        optimizer.zero_grad()
        
        outputs = model(inputs)
        
        loss = my_loss(outputs, targets)
        loss.backward()
        optimizer.step()
        
        LOSS.append(loss)
        
        
        
#%% LOSS PLOT
LOSS = torch.tensor(LOSS)
LOSS = LOSS.to('cpu')

plt.plot([i for i in range(len(LOSS))], LOSS)
plt.show()

#%% Validation = 98.51 percent

CP = 0 # Correct perdictions
Total = len(validation)

for example in validation:
    image = example[0][None, :].to(device)
    number = example[1]
    
    output = model(image)
    
    if output.argmax() == number:
        CP += 1
        
print(CP/Total * 100)


#%% Loading Test set

test_data = np.load(r'C:\Users\Korisnik\OneDrive\Desktop\Neural Networks Chalmers-Gothenburg\HW2\MNIST\MNIST\xTest2.npy')

Test = []
for i in range(10**4):
    test_image = (torch.tensor(test_data[:, :, 0, i])/256)[None, :].float()
    Test.append(test_image)
    
#%% Classifications = 98.75 percent 

Predictions = []

for case in Test:
    Predictions.append(int(model(case[None, :].to(device)).argmax()))

#%% Creating CSV file for an upload

df = pd.DataFrame(Predictions, columns = ['Classifications'])
df.to_csv(r'C:\Users\Korisnik\OneDrive\Desktop\Neural Networks Chalmers-Gothenburg\HW2\MNIST\Classifications.csv')







