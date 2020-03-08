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
        iSize= 128                                     # initial image size
        numRedLayers= 4                                # number of reducing layers
        fSize= int(iSize/(numRedLayers*red_stride))    # final size based on number of reducing layers and stride

        # fully connected layers
        self.fc1=nn.Linear(red_out*fSize*fSize, 512)
        self.fc2=nn.Linear(512, 1)

    # Feedforward function
    def forward(self, x):
        # 4 reducing convolution layers with 3 non-reducing layers
        print(x.size())
        r1 = func.relu(self.conv1_red(x))
        print(r1.size())
        n1 = func.relu(self.conv_non(r1))
        print(n1.size())
        r2 = func.relu(self.conv2_red(n1))
        print(r2.size())
        n2 = func.relu(self.conv_non(r2))
        print(n2.size())
        r3 = func.relu(self.conv2_red(n2))
        print(r3.size())
        n3 = func.relu(self.conv_non(r3))
        print(n3.size())
        r4 = func.relu(self.conv2_red(n3))
        print(r4.size())

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
        self.train
        inputs= torch.Tensor(image)
        targets= torch.Tensor(energy)
        outputs= self(inputs)
        obj_val= loss(self.forward(inputs), targets)
        optimizer.zero_grad()
        obj_val.backward()
        optimizer.step()
        return obj_val.item()

    # Test function. Avoids calculation of gradients.
    def test(self, image, energy, loss):
        self.eval()
        with torch.no_grad():
            inputs= torch.Tensor(image)
            targets= torch.Tensor(energy)
            outputs= self(inputs)
            cross_val= loss(self.forward(inputs), targets)
        return cross_val.item()