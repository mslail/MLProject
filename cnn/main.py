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
from data_obj import Data

sns.set_style("darkgrid")

if __name__ == '__main__':
    
    # Hyperparameters from json file
    jsonPath = sys.argv[1]
    with open(jsonPath) as json_file:
        param = json.load(json_file)
        
    # Getting excel data
    csvFile = "even_mnist.csv"
    data = np.array(list(csv.reader(open(csvFile, "r"), delimiter=" "))).astype("float")
    
    # reshaping images and isolating labels
    images = []
    labels = []
    
    for img in data:
        images.append(np.reshape(img[:-1], (1,14,14)))
        labels.append(img[-1])
    
    # separating into training and test sets
    trainImg = np.array(images[:26492], dtype=np.float32)
    testImg = np.array(images[26492:], dtype=np.float32)
    trainLbl = np.array(labels[:26492]).transpose()
    testLbl = np.array(labels[26492:]).transpose()

    # creating data structure
    data = Data(trainImg, testImg, trainLbl, testLbl)

    # constructing a model - assuming square image - passing in kernel size for convolution, and maxpool and image size
    model = Net(param["conv_kernel"], param["pool_kernel"], param["im_size"])
    
    # declare optimizer and gradient and loss function
    lr_rate = param["learning_rate"]

    optimizer = optim.SGD(model.parameters(), lr=lr_rate)
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

        train_val= model.backprop(data, loss, epoch, optimizer)     # training loss value
        obj_vals.append(train_val)
        
        test_val= model.test(data, loss, epoch)                     # test data loss value
        cross_vals.append(test_val)
        
        if not ((epoch + 1) % disp_epochs):
            print('Epoch [{}/{}]'.format(epoch+1, num_epochs)+\
              '\tTraining Loss: {:.4f}'.format(train_val)+\
              '\tTest Loss: {:.4f}'.format(test_val))
