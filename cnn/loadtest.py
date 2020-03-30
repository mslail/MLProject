import json, os, argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from cnn import Net
from main import convertToNumpy, loadimg

if __name__ == '__main__':

    # Command line arguments
    parser = argparse.ArgumentParser(description='Getting data from user')
    parser.add_argument('-o', metavar='dirName', type=str, default= "plot",
                        help='directory to store plots and other results of testing model')
    parser.add_argument('-i', metavar='imFile', type=str,
                        help='image file containing set of images and energy eigenvalues') 
    parser.add_argument('-m', metavar='modelFile', type=str,
                        help='file holding the current model') 
    parser.add_argument('-g', metavar='enableCuda', type=int, default=0,
                        help='enter 0 to run without GPU; 1 to run with GPU enabled')
    parser.add_argument('-j', metavar='jsonFile', type=str, default="parameters.json",
                        help='name of json file containing all hyperparameters')
    parser.add_argument('-v', metavar='verbosity', type=str, default=0,
                        help='verbosity - 0 or 1 ')
                                       
    args = parser.parse_args()

    # Getting inputs from cmd line
    outdir = str(args.o)+"/"
    imFile = str(args.i)
    modelFile = str(args.m)
    enableCuda = bool(args.g)
    vb = int(args.v)
    jsonPath = str(args.j)
    
    # creating output directory if it does not exist
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    
    # hyperparameters (json file) import
    with open(jsonPath) as json_file:
        param = json.load(json_file)

    # reshaping data and identifying labels
    images, energies = loadimg(imFile)
    print("Successfully loaded image data")
        # print details if verbosity is 1
    if vb == 1:
        print("Name of file: ", imFile,
              " Size of sample file: ", np.shape(images))

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
                nonred_stride=nStride, red_out=rOut, nonred_out=nOut,
                d="cuda:0" if enableCuda else "cpu")
    
    model.double()
    if enableCuda:
        model.cuda()

    # declare optimizer and gradient and loss function
    optimizer = optim.Adadelta(model.parameters(), lr=lr_rate)
    loss = torch.nn.MSELoss(reduction='mean')

    print("Loading model")
    model.load_state_dict(torch.load(modelFile))

    print("Starting Testing")
    out_nrgs, test_val = model.test(images, energies, loss)
    print("Testing Finished")

    out_nrgs = convertToNumpy(out_nrgs, enableCuda)
    # scaling energies to mHa
    energies = 1000*energies
    out_nrgs = 1000*out_nrgs
    
    # calculating median absolute error for energies in range (100-400 mHa)
    abs_err = abs(np.sort(energies) - np.sort(out_nrgs))
    median_err = np.median(abs_err)

    with open(outdir + "median_abs_error.txt","w+") as f:
        f.write("Median Absolute Error: " + str(median_err) + " mHa \n")
    
    # plotting true energies vs. predicted energies 
    plt.plot(np.linspace(100, max(energies)), np.linspace(100, max(energies)), linewidth=1.5, color='k')
    plt.scatter(np.sort(energies), np.sort(out_nrgs), s=1.5, color='red')
    plt.xlabel("True Energy (mHa)")
    plt.ylabel("Predicted Energy (mHa)")
    plt.xlim((100,400))
    plt.ylim((100,400))       

    plt.savefig(outdir + modelFile.split("\\")[-1] +
                "-data=" + imFile.replace("npy", "").replace("\\", "")[1:] + "png")
    plt.close()
