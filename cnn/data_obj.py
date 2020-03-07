"""

PHYS 490 
Assignment 2
Rubin Hazarika (20607919)

"""

import numpy as np

class Data():
    def __init__(self, imTrain, imTest, lblTrain, lblTest):
        """ Data generation
    im*  images for training or test dataset
    lbl* labels for training or test dataset """
        
        self.imTrain= imTrain
        self.imTest= imTest
        self.lblTrain= lblTrain
        self.lblTest= lblTest