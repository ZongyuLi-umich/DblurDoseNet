# DblurDoseNet

https://github.com/ZongyuLi-umich/DblurDoseNet

PyTorch implementation of DblurDoseNet
method for SPECT dosimetry,
as described in the paper
"DblurDoseNet: A deep residual learning network for voxel radionuclide dosimetry compensating for SPECT imaging resolution",
by
Zongyu Li, Jeffrey A Fessler, Justin K Mikell, Scott J Wilderman, Yuni K Dewaraja;
Medical Physics 49(2):1216-30, Feb. 2021.
https://doi.org/10.1002/mp.15397

# Dataset 

https://doi.org/10.7302/ykz6-cn05

# Training
`python3 train.py --batch [batch size] --lr [learning rate] --epochs [# of epochs]` <br>
For example, with batch size set to 32, learning rate set to 0.002, epochs number set to 200, 
the training command is <br>
`python3 train.py --batch 32 --lr 0.002 --epochs 200`

# Testing
To use the best checkpoint (the ckpt having the lowest validation loss), run the following command: <br>
`python3 test.py --batch [batch size] --is_best`
