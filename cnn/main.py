# use > conda activate base (in terminal)

import os, argparse, json
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from cnn import Net

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
    
    # Command line arguments
    parser = argparse.ArgumentParser(description='Getting data from user')
    parser.add_argument('-o', metavar='dirName', type=str, default="plot",
                        help='directory to store plots and other results of testing model')
    parser.add_argument('-i', metavar='imDir', type=str, default="samples",
                        help='directory containing set of image and energy eigenvalue files') 
    parser.add_argument('-d', metavar='modelSaveDir', type=str, default="models",
                        help='directory to store models') 
    parser.add_argument('-g', metavar='enableCuda', type=str, default=0,
                        help='enter 0 to run without GPU; 1 to run with GPU enabled')
    parser.add_argument('-j', metavar='jsonFile', type=str, default="parameters.json",
                        help='name of json file containing all hyperparameters')
    parser.add_argument('-v', metavar='verbosity', type=str, default=0,
                        help='verbosity - 0 or 1 ')
    args = parser.parse_args()
    
    # get image dir name from cmd line
    outdir = str(args.o)+"/"
    imdir = str(args.i)
    moddir = str(args.d)+"/"
    enableCuda = bool(args.g)
    vb = int(args.v)
    jsonPath = str(args.j)
    
    files = os.listdir(imdir)
    # creating output directories for if they does not exist
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    if not os.path.exists(moddir):
        os.mkdir(moddir)
    
    # hyperparameters (json file) import
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
    t_img, t_nrg = loadimg(imdir + "\\" + files[-1])
    
    # Training loop
    print("Attempting to Start training")
    for f in files[:-1]:
        img, nrg = loadimg(imdir + "\\" + f)
        for epoch in range(0, num_epochs):
            train_val = model.backprop(img, nrg, loss, optimizer)        
            # appending loss values for training dataset
            obj_vals.append(train_val)

            if not ((epoch + 1) % disp_epochs):
                print("file name:{}".format(f) +
                      "\tEpoch [{}/{}]".format(epoch+1, num_epochs) +
                      "\tTraining Loss: {:.5f}".format(train_val))

    print("Training Finished")
    
    pt = files[0][files[0].index("_"):files[0].index("_")+4]
    tsc = files[0][(files[0].index(" ")+1)+files[0][files[0].index(" ")+1:].index(" ")+1:files[0].index("]")]
    print("Saving Model")
    torch.save(model.state_dict(), moddir +
               "model_{}{}".format(pt,tsc))

    print("Starting Testing")
    out_nrgs, test_val = model.test(t_img, t_nrg, loss)
    print("Testing Finished")

    out_nrgs = convertToNumpy(out_nrgs, enableCuda)

    plt.plot(np.linspace(100, 400),
             np.linspace(100, 400), color="black")
    plt.scatter(np.sort(t_nrg)*1000, np.sort(out_nrgs)*1000, s=0.5, c='#FF0000')
    plt.xlim((100,400))
    plt.ylim((100,400))  
    plt.savefig(outdir + "InitialE.png")
    plt.close()

    plt.plot(obj_vals)
    plt.savefig(outdir + "Training error.png")
    plt.close()
