# This script consumes a data set of images (2D numpy array) and return a rotated set using the user-specified angle of rotation. 

import sys
import json, argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim


if __name__ == '__main__':

    # Command line arguments
    parser = argparse.ArgumentParser(description='Getting data from user')
    parser.add_argument('-d', metavar='imDataset', type=str, nargs=1,
                        help='.npy file containing images to rotate')
    parser.add_argument('-r', metavar='numRotate', type=str, nargs=1, default= [1],
                        help='number of 90 degree rotations to perform (1, 2, 3') 
    parser.add_argument('-g', metavar='enableCuda', type=str, nargs=1, default=[0],
                        help='enter 0 to run without GPU; 1 to run with GPU enabled')
                                       
    args = parser.parse_args()

    # Getting inputs from cmd line
    imFile= str(args.d[0])
    enableCuda= int(args.g[0])
    numRotate= int(args.r[0])

    # Reading in file 
    imData = np.load(imFile, allow_pickle=True)
    imLen = int(np.sqrt(len(imData[0]) - 1))     # pixel length of square image
    
    # Arrays to store rotated images  
    images = []

    # storing energies and images in arrays
    for img in imData:
        energy = img[-1]                                    # storing energy eigenvalue for later 
        imPre = np.reshape(img[:-1], (imLen, imLen))        # reshaping into 128x128 image before rotation
        imPost = np.rot90(imPre, k=numRotate)               # rotating image by 90 numRotate times
        imPostShape = np.reshape(imPost,(imLen*imLen))      # reshaping into flat array
        
        imPostFinal = np.append(imPostShape, energy)        # appending energy to flattened image
          
        images.append(imPostFinal)                          # appending to final dataset array

    # saving file as npy array with same name (with _rotated_(numRotate_90))
    filename = imFile + "_rotated_" + str(numRotate) + "X"
    np.save(filename, np.array(images))