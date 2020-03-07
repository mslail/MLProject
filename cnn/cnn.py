"""

PHYS 490 
Assignment 2
Rubin Hazarika (20607919)

"""

import torch
import torch.nn as nn
import torch.nn.functional as func

class Net(nn.Module):
    '''
    Neural network class.
    Architecture:
        Two convolution functions (convl1, convl2)
        One nonlinear function relu
        One maxpool function
        Three fully-connected layers fc1, fc2 and fc3.

    '''

    def __init__(self, cKSize, pKSize, imSize):
        super(Net, self).__init__()
        self.convl1= nn.Conv2d(1,3, kernel_size= cKSize)
        self.convl2= nn.Conv2d(3,10, kernel_size= cKSize)
        self.maxPool= nn.MaxPool2d(pKSize, stride= pKSize)        

        sizeC1 = (imSize - cKSize) + 1               # size after 1st convolution
        sizeC2 = (sizeC1 - cKSize) + 1               # size after 2nd convolution
        sizeP1 = int((sizeC2 - pKSize)/pKSize + 1)   # size after pooling 

        self.fc1= nn.Linear(10*sizeP1*sizeP1, 100)
        self.fc2= nn.Linear(100,30)
        self.fc3= nn.Linear(30,10)

    # Feedforward function
    def forward(self, x):
        # 1 iteration of convolution, relu and pooling
        l1 = func.relu(self.convl1(x))
        l1 = self.maxPool(func.relu(self.convl2(l1)))

        y = l1.view(-1, l1.size(1)*l1.size(2)*l1.size(3))  # flattening before applying linear functions
        y = func.relu(self.fc1(y))
        y = func.relu(self.fc2(y))
        y = self.fc3(y)
        return y

    # Reset function for the training weights
    # Use if the same network is trained multiple times.
    def reset(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

    # Backpropagation function
    def backprop(self, data, loss, epoch, optimizer):
        self.train()
        inputs= torch.tensor(data.imTrain)
        targets= torch.tensor(data.lblTrain, dtype = torch.long)
        outputs= self(inputs)
        obj_val= loss(self.forward(inputs), targets)
        optimizer.zero_grad()
        obj_val.backward()
        optimizer.step()
        return obj_val.item()

    # Test function. Avoids calculation of gradients.
    def test(self, data, loss, epoch):
        self.eval()
        with torch.no_grad():
            inputs= torch.tensor(data.imTest)
            targets= torch.tensor(data.lblTest, dtype = torch.long)
            outputs= self(inputs)
            cross_val= loss(self.forward(inputs), targets)
        return cross_val.item()
