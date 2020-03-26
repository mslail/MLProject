import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from cnn import Net
from main import convertToCpu, convertToTensor


vb = 0
dirName = "plot/"

if __name__ == '__main__':

    # get image file name from cmd line and load in pickled file
    imFile = sys.argv[1]
    modelFile = sys.argv[2]
    enableCuda = int(sys.argv[3])
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

    images = convertToTensor(images, torch.double, enableCuda)
    energies = convertToTensor(energies, torch.double, enableCuda)

    if enableCuda:
        model.to('cuda')

    # declare optimizer and gradient and loss function
    optimizer = optim.Adadelta(model.parameters(), lr=lr_rate)
    loss = torch.nn.MSELoss(reduction='mean')

    print("Loading model")
    model.load_state_dict(torch.load(modelFile))

    print("Starting Testing")
    out_nrgs, test_val = model.test(images, energies, loss)
    print("Testing Finished")

    energies = convertToCpu(energies, enableCuda)
    images = convertToCpu(images, enableCuda)
    out_nrgs = convertToCpu(out_nrgs, enableCuda)

    plt.plot(np.linspace(0.0, max(energies)),
             np.linspace(0.0, max(energies)))
    plt.scatter(np.sort(energies), np.sort(
        out_nrgs.numpy()), s=0.5)
    plt.savefig(dirName + modelFile.split("\\")[-1] +
                "-data=" + imFile.replace("npy", "").replace("\\", "")[1:] + "png")
    plt.close()
