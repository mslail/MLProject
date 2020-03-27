import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from cnn import Net
from main import convertToCpu, convertToTensor

if __name__ == '__main__':

    # Command line arguments
    parser = argparse.ArgumentParser(description='Getting data from user')
    parser.add_argument('-o', metavar='dirName', type=str, nargs=1, default= ["plot/"],
                        help='directory to store plots and other results of testing model')
    parser.add_argument('-i', metavar='imFile', type=str, nargs=1, default= ["sample.npy"],
                        help='image file containing set of images and energy eigenvalues') 
    parser.add_argument('-m', metavar='modelFile', type=str, nargs=1,
                        help='file holding the current model') 
    parser.add_argument('-g', metavar='enableCuda', type=str, nargs=1, default=[0],
                        help='enter 0 to run without GPU; 1 to run with GPU enabled')
    parser.add_argument('-j', metavar='jsonFile', type=str, nargs=1, default=["parameters.json"],
                        help='name of json file containing all hyperparameters')
    parser.add_argument('-v', metavar='verbosity', type=str, nargs=1, default=[0],
                        help='verbosity - 0 or 1 ')
                                       
    args = parser.parse_args()

    # Getting inputs from cmd line
    dirName = str(args.o[0])
    imFile = str(args.i[0])
    modelFile = str(args.m[0])
    enableCuda = int(args.g[0])
    vb = int(args.v[0])

    # loading in image data
    imData = np.load(imFile, allow_pickle=True)

    # creating directory for results if it does not exist
    if not os.path.exists(dirName):
        os.mkdir(dirName)

    # print details if verbosity is 1
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

    # formatting all output
    energies = convertToCpu(energies, enableCuda)           # true values for energies
    images = convertToCpu(images, enableCuda)               # images 
    out_nrgs = convertToCpu(out_nrgs, enableCuda)           # predicted values for energies

    # scaling energies to mHa
    energies *= 1000
    out_nrgs *= 1000

    # plotting true energies vs. predicted energies 
    plt.plot(np.linspace(0.0, max(energies)), np.linspace(0.0, max(energies)))
    plt.scatter(np.sort(energies), np.sort(out_nrgs.numpy()), s=0.5)
    plt.xlabel("True Energy (mHa)")
    plt.ylabel("Predicted Energy (mHa)")
    plt.xlim((100,400))
    plt.ylim((100,400))       

    plt.savefig(dirName + modelFile.split("\\")[-1] +
                "-data=" + imFile.replace("npy", "").replace("\\", "")[1:] + "png")
    plt.close()
