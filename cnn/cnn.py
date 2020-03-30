import torch
import torch.nn as nn
import torch.nn.functional as func

class Net(nn.Module):

    # INPUTS:
    # [...]_kernel = kernel size for the convolution layers
    # [...]_stride = stride for the convolution layers
    # [...]_out = # of output channels for reducing vs. non-reducing layers

    def __init__(self, red_kernel, nonred_kernel, red_stride, nonred_stride, red_out, nonred_out, d):
        super(Net, self).__init__()
        self.dev = d
        # convolution layers: '[...]_red' is a reducing layer; '[...]_non' is a non-reducing layer
        self.conv1_red = nn.Conv2d(
            1, red_out, kernel_size=red_kernel, stride=red_stride)
        self.conv2_red = nn.Conv2d(
            nonred_out, red_out, kernel_size=red_kernel, stride=red_stride)

        self.conv_non = nn.Conv2d(
            red_out, nonred_out, kernel_size=nonred_kernel, stride=nonred_stride)

        # size variables based on number of layers
        fSize = 4    # final size based on number of reducing layers and stride

        # fully connected layers
        self.fc1 = nn.Linear(red_out*fSize*fSize, 1024)
        self.fc2 = nn.Linear(1024, 1)

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
        # flattening before applying linear functions
        h = h.view(-1, h.size(1)*h.size(2)*h.size(3))
        h = func.relu(self.fc1(h))
        h = func.relu(self.fc2(h))

        return h

    # Reset function for the training weights
    # Use if the same network is trained multiple times.
    def reset(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

    # Backpropagation function
    def backprop(self, image, energy, loss, optimizer):
        self.train()
        # preparing input and target tensors with proper dtype
        inputs = torch.from_numpy(image).to(self.dev)
        targets = torch.from_numpy(energy).to(self.dev)

        # calculating output from nn and reshaping
        obj_val = loss(self.forward(inputs).view(-1), targets)

        optimizer.zero_grad()
        obj_val.backward()
        optimizer.step()
        return obj_val.item()

    # Test function. Avoids calculation of gradients.
    def test(self, image, energy, loss):
        self.eval()
        with torch.no_grad():
            inputs = torch.from_numpy(image).to(self.dev)
            targets = torch.from_numpy(energy).to(self.dev)

            # calculating output from nn and reshaping
            calculatedVal = self.forward(inputs).view(-1)
            cross_val = loss(calculatedVal, targets)

        return calculatedVal, cross_val.item()
