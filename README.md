# MLProject
Project for Phys 490.


# Running CNN

```sh
python main.py [sample_directorry] [enable_gpu]
```

- where sample_directory is the folder containing the .npy files with the images and energy eigenvalues
*Must be more that 1 .npy file* *# of samples per file does not matter*
- enable_gpu is either 1 or 0 if you want cuda enabled

# Loading and Running CNN

```sh
python loadtest.py sample.npy [path_to_model] [enable_gpu]
```

- where sample.npy is the numpy file containing the images and energy eigenvalues
- enable_gpu is either 1 or 0 if you want cuda enabled
- path_to_model is the path to model saved

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
