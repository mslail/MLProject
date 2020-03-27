"""

PHYS 490
Assignment 2
Rubin Hazarika (20607919)

"""
# use > conda activate base (in terminal)

import sys
import numpy as np
import json
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from cnn import Net

sns.set_style("darkgrid")

# verbosity (0,1,2) - for different levels of debugging
vb = 0

dirName = "plot/"
modelSaveDir = "models/"


def convertToTensor(obj, tensorType, cudaToggle):
    if cudaToggle:
        return torch.tensor(obj, dtype=tensorType).to('cuda')
    return torch.tensor(obj, dtype=tensorType)


def convertToCpu(tensorObj, cudaToggle):
    if cudaToggle:
        return tensorObj.cpu().detach()
    return tensorObj.detach()


if __name__ == '__main__':

    # get image file name from cmd line and load in pickled file
    imFile = sys.argv[1]
    enableCuda = int(sys.argv[2])
    imData = np.load(imFile, allow_pickle=True)
    if vb == 1:
        print("Name of file: ", imFile,
              " Size of sample file: ", np.shape(imData))

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
        images.append(np.reshape(img[:-1], (1, imLen, imLen)))
        energies.append(img[-1])

    print("Successfully loaded image data")

    # hyperparameters (start with r = for reducing layer, start with n = for nonreducing layer)
    lr_rate = float(param["learning_rate"])
    rKern = int(param["reducing_conv_kernel"])
    nKern = int(param["nonreducing_conv_kernel"])
    rStride = int(param["reducing_stride"])
    nStride = int(param["nonreducing_stride"])
    rOut = int(param["reducing_out"])
    nOut = int(param["nonreducing_out"])

    # constructing a model (converting model to double precision)
    model = Net(red_kernel=rKern, nonred_kernel=nKern, red_stride=rStride,
                nonred_stride=nStride, red_out=rOut, nonred_out=nOut)
    model = model.double()
    dataLen = len(images)

    if enableCuda:
        model.to('cuda')

    # declare optimizer and gradient and loss function
    optimizer = optim.Adadelta(model.parameters(), lr=lr_rate)
    loss = torch.nn.MSELoss(reduction='mean')

    # storing the training and test loss values
    obj_vals = []
    cross_vals = []

    # epoch hyperparameters
    epochs = param["num_epochs_per_batch"]
    disp_epochs = param["display_epochs"]
    num_epochs = int(epochs)
    n_batches = int(param["n_batches"])
    batch_size = int(param["batch_size"])

    # Checking for n_batches and batch_size
    if (n_batches >= dataLen) or (batch_size >= dataLen):
        raise Exception(
            "Error: ensure that n_batches({}) or batch_size({}) "
            "are within the data length({})".format(n_batches, batch_size, dataLen))

    # Sectioning off training and testing images
    test = convertToTensor(images[n_batches:], torch.double, enableCuda)
    testE = convertToTensor(energies[n_batches:],  torch.double, enableCuda)

    # Training loop
    print("Attempting to Start training")
    for batch in range(0, n_batches - batch_size, batch_size):
        batch_images = convertToTensor(
            images[batch: batch+batch_size], torch.double, enableCuda)
        batch_energies = convertToTensor(
            energies[batch: batch+batch_size], torch.double, enableCuda)
        for epoch in range(1, num_epochs + 1):
            output, train_val = model.backprop(
                batch_images, batch_energies, loss, optimizer)        # training loss value
            # appending loss values for training dataset
            obj_vals.append(train_val)

            if not ((epoch + 1) % disp_epochs):
                output = convertToCpu(output, enableCuda).numpy()

                plt.plot(output)
                plt.savefig(dirName + "E" + str(epoch) + ".png")
                plt.close()
                print("Batch #{}".format(int(batch/batch_size)) +
                      "\tEpoch [{}/{}]".format(epoch, num_epochs) +
                      "\tTraining Loss: {:.6f}".format(train_val))

    print("Training Finished")
    
    # Saving model
    print("Saving Model")
    torch.save(model.state_dict(), modelSaveDir +
               "model=file-{}".format(imFile.replace("npy", "").replace("\\", "")[1:]))

    # Not preliminary testing
    print("Starting Testing")

    out_nrgs, test_val = model.test(test, testE, loss)
    print("Testing Finished")

    testE = convertToCpu(testE, enableCuda)
    test = convertToCpu(test, enableCuda)
    out_nrgs = convertToCpu(out_nrgs, enableCuda)

    plt.plot(np.linspace(0.0, max(testE)),
             np.linspace(0.0, max(testE)))
    plt.scatter(np.sort(testE), np.sort(
        out_nrgs.numpy()), s=0.5)
    plt.savefig(dirName + "InitialE.png")
    plt.close()

    plt.plot(obj_vals)
    plt.savefig(dirName + "Training error.png")
    plt.close()

    plt.plot(cross_vals)
    plt.savefig(dirName + "Test error.png")
    plt.close()
