"""

PHYS 490 
Assignment 2
Rubin Hazarika (20607919)

"""
# use > conda activate base (in terminal)

import sys
import numpy as np
import json, argparse, torch
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from cnn import Net

sns.set_style("darkgrid")

# verbosity (0,1,2) - for different levels of debugging
vb = 0

if __name__ == '__main__':
    
    # get image file name from cmd line and load in pickled file
    imFile = sys.argv[1]
    imData = np.load(imFile, allow_pickle=True)   
    if vb == 1: print("Name of file: ", imFile, " Size of sample file: ", np.shape(imData))
   
    # hyperparameters (json file) import
    jsonPath = "parameters.json"
    with open(jsonPath) as json_file:
        param = json.load(json_file)
    
    # reshaping data and identifying labels
    imLen = int(np.sqrt(len(imData[0]) - 1))     # pixel length of square image
    images = []
    energies = []

    # storing energies and images in arrays
    for img in imData:
       images.append(np.reshape(img[:-1], (1,imLen,imLen)))
       energies.append(img[-1])
    
    # hyperparameters
    lr_rate = int(param["learning_rate"])
    rKern = int(param["reducing_conv_kernel"])
    nKern = int(param["nonreducing_conv_kernel"])
    rStride = int(param["reducing_stride"])
    nStride = int(param["nonreducing_stride"])
    rOut = int(param["reducing_out"])
    nOut = int(param["nonreducing_out"])

    # constructing a model
    model = Net(red_kernel=rKern, nonred_kernel=nKern, red_stride=rStride, nonred_stride=nStride, red_out=rOut, nonred_out=nOut)
    
    # declare optimizer and gradient and loss function
    optimizer = optim.Adadelta(model.parameters(), lr=lr_rate)
    loss = torch.nn.CrossEntropyLoss(reduction= 'mean')
    
    # storing the training and test loss values
    obj_vals= []
    cross_vals= []
    
    # epoch hyperparameters
    epochs = param["num_epochs"] 
    disp_epochs = param["display_epochs"]
    num_epochs= int(epochs)

    # Training loop
    for epoch in range(1, num_epochs + 1):

        train_val= model.backprop(images, energies, loss, optimizer)     # training loss value
        obj_vals.append(train_val)

        if not ((epoch + 1) % disp_epochs):
            print('Epoch [{}/{}]'.format(epoch+1, num_epochs)+\
              '\tTraining Loss: {:.4f}'.format(train_val))
