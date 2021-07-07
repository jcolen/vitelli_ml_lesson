# Vitelli Group ML lesson

This repository defines a basic ML training procedure for demonstration purposes. 
Three models are implemented for the purposes of identifying defects within a given fluorescence image.
They can be trained by running `train_nn.py`. The training will default to 1000 epochs, but this can be changed with the `--max_epochs` flag. 

Training is logged in Tensorboard file to a directory `tb_logs`. To view this, run `tensorboard --logdir tb_logs` and open `localhost:6006` in a browser. 

## Requirements

- numpy
- matplotlib
- pandas
- torch
- torchvision
- pytorch-lightning
- tensorboard
- scipy
- scikit-image
- numba

## Contact

Please email jcolen@uchicago.edu with any questions or concerns.
