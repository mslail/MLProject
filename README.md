# MLProject
Project for Phys 490.

# Running the Code
 There are 3 modular components to performing the analysis done in the paper:
 
 1. Generating 2D potential images
 2. Training model on sample images
 3. Testing model on new images

## 1. Generating 2D potential images

This section of code, found in (data_generation/) can be run using:

```sh
python Gen_main.py [n_samples] [Potential type]
```
-where n_samples is the total number of samples generated and Potential_type is the 3 letter code for
the desired potential. Avalible potentials are the simple harmonic oscilator (HMO) and infinite potential well (IPW)

-optional parameters include number of samples per file (-n_per: default 1000), switch to use known analytic solution
for energy values (-use_nrg: default true), and the image resolution (-res: default 128)

**Output:** 
- the output of the program will be .npy image files containing 128 x 128 images
- there are pre-made image sets in the folder (cnn/test_data) containing images for the simple harmonic oscillator (SHO) and inifinite well (IPW)

## 2. Training model on sample images

This section of code, found in (cnn/), can be run using:

```sh
python main.py -o [dirName] -i [imDir] -d [modelSaveDir] -g [enableCuda] -j [jsonFile] -v [verbosity]
```
- all the user-modifiable parameters have default settings allowing for this code to run without user-input
- imDir is the folder containing the .npy files with the images and energy eigenvalues (*Must be more that 1 .npy file* *# of samples per file does not matter*)

**Output:** 
- this script trains a model using the neural network defined in *cnn.py* and stores the model in the modelSaveDir directory. 
- there are already pre-made models (for the simple harmonic oscillator) for user access in the folder (cnn/models/)

## 3. Testing model on new images 

```sh
python loadtest.py -i imFile -o [dirName] -m [modelFile] -g [enableCuda] -j [jsonFile] -v [verbosity]
```

- where imFile (the .npy image set for testing) is an essential input (all others have defaults)

**Output:**
- this script will run the model on the specified image dataset and produce a histogram of true vs. predicted energies
- it will also calculate the median absolute error as a means of gauging the accuracy of the model on the given dataset

# Parameters (Sample, replace () with desired values) 
```
{
	"learning_rate": (Learning rate used for training),
	"num_epochs_per_batch": (Number of epochs ran per batch),
	"display_epochs": (Display epoch, shows loss at each step),
	"reducing_conv_kernel": (Convolution kernel),
	"reducing_stride": (Stride),
	"nonreducing_conv_kernel": (Non reducing convolution kernel),
	"nonreducing_stride": (Non reducing stride),
	"reducing_out": (Reducing out),
	"nonreducing_out": (Non reducing out)
}
```

# Changes from Paper and HW2

1) 128 x 128 image (versus 256 x 256 expected in paper)
2) 4 reducing layers with 1 non-reducing in between (instead of 7 reducing layers with 2 non-reducing in between)
3) Removed data object used in HW2 assignment 
4) 16 output channels on reducing layer (64 in paper) and 4 output channels for non-reducing layer (16 in paper)
5) Fully connected layers have outputs of 512 (1024 in paper) and 1
6) Adadelta Optimizer used (as opposed to SGD in HW2)
7) MSELoss used (as opposed to CrossEntropyLoss in HW2) 
