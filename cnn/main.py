"""

PHYS 490 
Assignment 2
Rubin Hazarika (20607919)

"""
# use > conda activate base (in terminal)

import numpy as np
import json, argparse, torch
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from cnn import Net

sns.set_style("darkgrid")

# verbosity (0,1,2) - for different levels of debugging
vb = 0

dirName = "plot/"

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser("Classify even dataset")
    parser.add_argument('indir', metavar='i', help='name directory for input')
    parser.add_argument('-g', default=0, type=bool, help="enable gpu usage")
    args = parser.parse_args()
    
    # get image file name from cmd line and load in pickled file
    imFile = args.indir
    imData = np.load(imFile, allow_pickle=True)
    np.random.shuffle(imData)
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
    
    # hyperparameters (start with r = for reducing layer, start with n = for nonreducing layer)
    lr_rate = float(param["learning_rate"])
    rKern = int(param["reducing_conv_kernel"])
    nKern = int(param["nonreducing_conv_kernel"])
    rStride = int(param["reducing_stride"])
    nStride = int(param["nonreducing_stride"])
    rOut = int(param["reducing_out"])
    nOut = int(param["nonreducing_out"])

    # constructing a model (converting model to double precision)
    model = Net(red_kernel=rKern, nonred_kernel=nKern, red_stride=rStride, nonred_stride=nStride, red_out=rOut, nonred_out=nOut,\
                d = "cuda:0" if args.g else "cpu")
    model = model.double()
    if args.g:
        model.cuda()
    
    # declare optimizer and gradient and loss function
    optimizer = optim.Adadelta(model.parameters(), lr=lr_rate)
    loss = torch.nn.MSELoss(reduction= 'mean')
    
    # storing the training and test loss values
    obj_vals= []
    cross_vals= []
    out_nrgs = []
    
    # epoch hyperparameters
    epochs = param["num_epochs"] 
    disp_epochs = param["display_epochs"]
    num_epochs= int(epochs)

    images1 = images[100:]
    test = images[:100]
    energies1= energies[100:] 
    testE = energies[:100]

    # Training loop
    for epoch in range(1, num_epochs + 1):
        train_val= model.backprop(images1, energies1, loss, optimizer)        # training loss value
        obj_vals.append(train_val)                                                  # appending loss values for training dataset     

        out_nrgs, test_val= model.test(test, testE, loss)  
        cross_vals.append(test_val)

        if not ((epoch) % disp_epochs):
            print('Epoch [{}/{}]'.format(epoch, num_epochs)+\
              '\tTraining Loss: {:.4f}'.format(train_val) + \
               '\t Test Loss: {:.4f}'.format(test_val))
    
    out_nrgs, test_val= model.test(images1, energies1, loss)
    
    plt.plot(np.linspace(0.0,max(energies1)),np.linspace(0.0,max(energies1)))
    plt.scatter(np.sort(energies1),np.sort(out_nrgs.detach().cpu().numpy()), s=0.5)
    plt.savefig(dirName + "InitialE.png") 
    plt.close()

    plt.plot(obj_vals)
    plt.savefig(dirName + "Training error.png") 
    plt.close()

    plt.plot(cross_vals)
    plt.savefig(dirName + "Test error.png") 
    plt.close()
 