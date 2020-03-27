"""

PHYS 490
Assignment 2
Rubin Hazarika (20607919)

"""
# use > conda activate base (in terminal)

import sys, os
import numpy as np
import json
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from cnn import Net

sns.set_style("darkgrid")

# verbosity (0,1,2) - for different levels of debugging
vb = 0

dirName = "plot/"
modelSaveDir = "models/"

def convertToNumpy(tensorObj, cudaToggle):
    if cudaToggle:
        return tensorObj.cpu().detach().numpy()
    return tensorObj.detach().numpy()

def loadimg(filename):
    D = np.load(filename, allow_pickle=True)
    
    # reshaping data and identifying labels
    imLen = int(np.sqrt(len(D[0]) - 1))     # pixel length of square image
    images = []
    energies = []

    # storing energies and images in arrays
    for img in D:
        images.append(np.reshape(img[:-1], (1, imLen, imLen)))
        energies.append(img[-1])
        
    return np.array(images), np.array(energies)

if __name__ == '__main__':

    # get image dir name from cmd line
    direc = sys.argv[1]
    files = os.listdir(direc)
    enableCuda = bool(sys.argv[2])
    
    # hyperparameters (json file) import
    jsonPath = "parameters.json"
    with open(jsonPath) as json_file:
        param = json.load(json_file)

    # hyperparameters (start with r = for reducing layer, start with n = for nonreducing layer)
    lr_rate = float(param["learning_rate"])
    rKern = int(param["reducing_conv_kernel"])
    nKern = int(param["nonreducing_conv_kernel"])
    rStride = int(param["reducing_stride"])
    nStride = int(param["nonreducing_stride"])
    rOut = int(param["reducing_out"])
    nOut = int(param["nonreducing_out"])
    
    # epoch hyperparameters
    num_epochs = int(param["num_epochs_per_batch"])
    disp_epochs = param["display_epochs"]

    # constructing a model (converting model to double precision)
    model = Net(red_kernel=rKern, nonred_kernel=nKern, red_stride=rStride,
                nonred_stride=nStride, red_out=rOut, nonred_out=nOut,
                d="cuda:0" if enableCuda else "cpu")
    model.double()
    if enableCuda:
        model.cuda()

    # declare optimizer and gradient and loss function
    optimizer = optim.Adadelta(model.parameters(), lr=lr_rate)
    loss = torch.nn.MSELoss(reduction='mean')

    # storing the training and test loss values
    obj_vals = []
    
    #load test dataset
    t_img, t_nrg = loadimg(direc + "\\" + files[-1])
    
    # Training loop
    print("Attempting to Start training")
    for f in files[:-1]:
        img, nrg = loadimg(direc + "\\" + f)
        for epoch in range(0, num_epochs):
            train_val = model.backprop(img, nrg, loss, optimizer)        
            # appending loss values for training dataset
            obj_vals.append(train_val)

            if not ((epoch + 1) % disp_epochs):
                print("file name:{}".format(f) +
                      "\tEpoch [{}/{}]".format(epoch+1, num_epochs) +
                      "\tTraining Loss: {:.5f}".format(train_val))

    print("Training Finished")
    # Saving model
    print("Saving Model")
    torch.save(model.state_dict(), modelSaveDir +
               "model_{}".format(files[0].replace("npy", "").replace("\\", "")[1:]))

    print("Starting Testing")

    out_nrgs, test_val = model.test(t_img, t_nrg, loss)
    print("Testing Finished")

    out_nrgs = convertToNumpy(out_nrgs, enableCuda)

    plt.plot(np.linspace(0.0, max(t_nrg)),
             np.linspace(0.0, max(t_nrg)))
    plt.scatter(np.sort(t_nrg), np.sort(out_nrgs), s=0.5)
    plt.savefig(dirName + "InitialE.png")
    plt.close()

    plt.plot(obj_vals)
    plt.savefig(dirName + "Training error.png")
    plt.close()
