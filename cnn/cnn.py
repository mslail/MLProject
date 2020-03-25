import torch
import torch.nn as nn
import torch.nn.functional as func

class Net(nn.Module):

    # INPUTS: 
    # [...]_kernel = kernel size for the convolution layers
    # [...]_stride = stride for the convolution layers
    # [...]_out = # of output channels for reducing vs. non-reducing layers

    def __init__(self, red_kernel, nonred_kernel, red_stride, nonred_stride, red_out, nonred_out, d="cpu"):
        super(Net, self).__init__()
        self.d = d
        
        # convolution layers: '[...]_red' is a reducing layer; '[...]_non' is a non-reducing layer
        self.conv1_red = nn.Conv2d(1, red_out, kernel_size= red_kernel, stride= red_stride) 
        self.conv2_red = nn.Conv2d(nonred_out, red_out, kernel_size= red_kernel, stride= red_stride)

        self.conv_non = nn.Conv2d(red_out, nonred_out, kernel_size= nonred_kernel, stride= nonred_stride)
        self.conv_non2 = nn.Conv2d(nonred_out, nonred_out, kernel_size= nonred_kernel, stride= nonred_stride)

        # size variables based on number of layers
        fSize= 4    # final size based on number of reducing layers and stride

        # fully connected layers
        self.fc1=nn.Linear(16*fSize*fSize, 512)
        self.fc2=nn.Linear(512, 1)

    # Feedforward function
    def forward(self, x):
        # 4 reducing convolution layers with 3 non-reducing layers
        h = func.relu(self.conv1_red(x))
        h = func.relu(self.conv_non(h))
        h = func.relu(self.conv2_red(h))
        h = func.relu(self.conv_non(h))
        h = func.relu(self.conv2_red(h))
        h = func.relu(self.conv_non(h))
        h = func.relu(self.conv2_red(h))
        
        # fully connected layers
        h = h.view(-1, h.size(1)*h.size(2)*h.size(3))  # flattening before applying linear functions
        h = func.relu(self.fc1(h))
        h = func.relu(self.fc2(h))

        return h

    # Reset function for the training weights
    # Use if the same network is trained multiple times.
    def reset(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.conv1_red.reset_parameters()
        self.conv2_red.reset_parameters()
        self.conv_non.reset_parameters()
        self.conv_non2.reset_parameters()

    # Backpropagation function
    def backprop(self, image, energy, loss, optimizer):
        self.train()
        # preparing input and target tensors with proper dtype
        inputs= torch.tensor(image, dtype= torch.double, requires_grad=True)
        targets= torch.tensor(energy, dtype= torch.double, requires_grad=False)
        inputs = inputs.to(device=self.d)
        targets = targets.to(device=self.d)
        
        # calculating loss
        #print(self.forward(inputs).view(-1))
        #print(targets)
        #print()
        obj_val = loss(self.forward(inputs[:len(inputs)//2]).view(-1), targets[:len(inputs)//2])
        print(obj_val.size())
        obj_val += loss(self.forward(inputs[:len(inputs)//2]).view(-1), targets[:len(inputs)//2])
        print(obj_val.size())
        optimizer.zero_grad()
        obj_val.backward()
        optimizer.step()
   
        return obj_val.item()

    # Test function. Avoids calculation of gradients.
    def test(self, image, energy, loss):
        self.eval()
        with torch.no_grad():
            inputs= torch.tensor(image, dtype= torch.double)
            targets= torch.tensor(energy, dtype= torch.double)
            inputs = inputs.to(device=self.d)
            targets = targets.to(device=self.d)
            
            nrgs = self.forward(inputs).view(-1)
            cross_val= loss(nrgs, targets)

        return nrgs, cross_val.item()