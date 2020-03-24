import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np

class Net(nn.Module):

    # INPUTS: 
    # [...]_kernel = kernel size for the convolution layers
    # [...]_stride = stride for the convolution layers
    # [...]_out = # of output channels for reducing vs. non-reducing layers

    def __init__(self, red_kernel, nonred_kernel, red_stride, nonred_stride, red_out, nonred_out):
        super(Net, self).__init__()

        # convolution layers: '[...]_red' is a reducing layer; '[...]_non' is a non-reducing layer
        self.conv1_red = nn.Conv2d(1, red_out, kernel_size= red_kernel, stride= red_stride) 
        self.conv2_red = nn.Conv2d(nonred_out, red_out, kernel_size= red_kernel, stride= red_stride)

        self.conv_non = nn.Conv2d(red_out, nonred_out, kernel_size= nonred_kernel, stride= nonred_stride)

        # size variables based on number of layers
        fSize= 4    # final size based on number of reducing layers and stride

        # fully connected layers
        self.fc1=nn.Linear(red_out*fSize*fSize, 1024)
        self.fc2=nn.Linear(1024, 1)

    # Feedforward function
    def forward(self, x):
        # 4 reducing convolution layers with 3 non-reducing layers
        r1 = func.relu(self.conv1_red(x))
        n1 = func.relu(self.conv_non(r1))
        r2 = func.relu(self.conv2_red(n1))
        n2 = func.relu(self.conv_non(r2))
        r3 = func.relu(self.conv2_red(n2))
        n3 = func.relu(self.conv_non(r3))
        r4 = func.relu(self.conv2_red(n3))
        
        # fully connected layers
        y = r4.view(-1, r4.size(1)*r4.size(2)*r4.size(3))  # flattening before applying linear functions
        y = func.relu(self.fc1(y))
        y = func.relu(self.fc2(y))

        return y

    # Reset function for the training weights
    # Use if the same network is trained multiple times.
    def reset(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

    # Backpropagation function
    def backprop(self, image, energy, loss, optimizer):
        self.train()
        # preparing input and target tensors with proper dtype
        inputs= torch.tensor(image, dtype= torch.double)
        targets= torch.tensor(energy, dtype= torch.double)
   
        # calculating output from nn and reshaping
        calculatedVal = self.forward(inputs)
        size = calculatedVal.size()[0]
        calculatedVal = calculatedVal.view(size)

        # calculating loss
        obj_val= loss(calculatedVal, targets)
        
        optimizer.zero_grad()
        obj_val.backward()
        optimizer.step()
   
        return calculatedVal, obj_val.item()

    # Test function. Avoids calculation of gradients.
    def test(self, image, energy, loss):
        self.eval()
        with torch.no_grad():
            inputs= torch.tensor(image, dtype= torch.double)
            targets= torch.tensor(energy, dtype= torch.double)

            # calculating output from nn and reshaping
            calculatedVal = self.forward(inputs)
            size = calculatedVal.size()[0]
            calculatedVal = calculatedVal.view(size)

            cross_val= loss(calculatedVal, targets)

        return cross_val.item()