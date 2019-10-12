# Plane or bird?
Is it a bird? Is it a plane?

Well the age old question does not need to worry you any more. __ALL NEW__ AI
can now solve this problem for you!

## Requirements

## Documentation
----------------
# Data preparation
There are two main folders to this project. Data and plane-or-bird. Data
contains all the required data for the model to run. Clearly these are png files
and so some preparation is needed before they can be analysed. [data_clean.py()](plane-or-bird/data_clean.py) does this for you!

data_clean.py will import the png files and transform them into numpy arrays or in human language, turn them into files a computer can read (_numbers_). This file will turn any 32x32 rgb image into a format readable by the models used later.

In future versions there will be accessibility for the inclusion of other images of other sizes, however for now please keep them to 32x32 in rgb format.

# DenseNET
DenseNET is the name of a deep learning algorithm that produced unbelievable results with such little training data, in fact the model was originally created to perform well on this specific dataset; however the original dataset contained many more training images and classes, this is a simplified version.

Your own model can be trained by running the [dense_net.py](plane-or-bird/dense_net.py) file. Otherwise a model can be loaded by following instructions (insert instructions) from the pretrained weights file [deepCNN.h5](plane-or-bird/pretrained/deepCNN.h5). With these weights an accuracy of nearly 90% can be expected on unseen data on just 10 epochs, (with loss still decreasing and validation accuracy increasing) so these weights could be used as a starting point to an improved model.

_More updates soon._s
