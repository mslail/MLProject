# MLProject
Project for Phys 490.


# Running CNN

```sh
python main.py sample.npy [enable_gpu]
```

- where sample.npy is the numpy file containing the images and energy eigenvalues
- enable_gpu is either 1 or 0 if you want cuda enabled

# Changes from Paper and HW2

1) 128 x 128 image (versus 256 x 256 expected in paper)
2) 4 reducing layers with 1 non-reducing in between (instead of 7 reducing layers with 2 non-reducing in between)
3) Removed data object used in HW2 assignment 
4) 16 output channels on reducing layer (64 in paper) and 4 output channels for non-reducing layer (16 in paper)
5) Fully connected layers have outputs of 512 (1024 in paper) and 1
6) Adadelta Optimizer used (as opposed to SGD in HW2)
7) MSELoss used (as opposed to CrossEntropyLoss in HW2) 
