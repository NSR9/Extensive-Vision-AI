# Assignment 12 - The Dawn of the Transformers:

## Problem Statement:

1. Implement that [Spatial Transformer Code](https://brsoff.github.io/tutorials/intermediate/spatial_transformer_tutorial.html) for CIFAR10 and submit. Must have proper readme and must be trained for 50 Epochs.
2. describe using text and your drawn images, the classes in this [FILE](https://github.com/jeonsworld/ViT-pytorch/blob/main/models/modeling.py) (Links to an external site.):
 - Block
 - Embeddings
 - MLP
 - Attention
 - Encoder
 
## Spatial Transformers(STN's):

Spatial Transformer Network (STN), by Google DeepMind helps to crop out and scale-normalizes the appropriate region, which can simplify the subsequent classification task and lead to better classification performance

A Spatial Transformer Network (STN) is a learnable module that can be placed in a Convolutional Neural Network (CNN), to increase the spatial invariance in an efficient manner. Spatial invariance refers to the invariance of the model towards spatial transformations of images such as rotation, translation and scaling. Invariance is the ability of the model to recognize and identify features even when the input is transformed or slightly modified. Spatial Transformers can be placed into CNNs to benefit various tasks. One example is image classification.

**The working of Spatial Transformer Network on the Distorted MNIST dataset can be seen as follows:**
![image](https://user-images.githubusercontent.com/51078583/127387882-ba6dda8c-304c-47fd-a64f-723f074395cd.png)


A STN is majorly divided into 3 parts :
- Localisation Net
- Grid Generator
- Sampler

Which can be visiualized using the following image:

![image](https://user-images.githubusercontent.com/51078583/127382590-c1f9ed10-2964-4829-a1c7-67580c3cec2e.png)

### 1. Localization Network

It is a simple neural network with a few convolution layers and a few dense layers. It predicts the parameters of transformation as output. These parameters determine the angle by which the input has to be rotated, the amount of translation to be done, and the scaling factor required to focus on the region of interest in the input feature map.
 
### 2. Grid Generator

The transformation parameters predicted by the localization net are used in the form of an affine transformation matrix of size 2 x 3 for each image in the batch. An affine transformation is one which preserves points, straight lines and planes. Parallel lines remain parallel after affine transformation. Rotation, scaling and translation are all affine transformations.

![image](https://user-images.githubusercontent.com/51078583/127388037-68615834-1b48-44a7-92fb-4abb266df9d8.png)

Here, (xti,yti) are the target coordinates of the target grid in the output feature map, (xsi,ysi) are the input coordinates in the input feature map, and Aθ is the affine transformation matrix. T is the transformation and A is the matrix representing the affine transformation. θ11, θ12, θ21, θ22 determine the angle by which the image has to be rotated. θ13, θ23 determine the translations along width and height of the image respectively. Thus we obtain a sampling grid of transformed indices.


### 3. Sampler

This is the last part of the spatial transformer network. We have the input feature map and also the parameterized sampling grid with us now. To perform the sampling, we give the feature map U and sampling grid Tθ(G) as input to the sampler. The sampling kernel is applied to the source coordinates using the parameters θ and we get the output V.

![image](https://user-images.githubusercontent.com/51078583/127388329-bcc89a75-6558-439d-bd25-52ad90d72415.png)

## CIFAR10 with STN:
Implement that [Spatial Transformer Code](https://brsoff.github.io/tutorials/intermediate/spatial_transformer_tutorial.html) for CIFAR10 and submit. Must have proper readme and must be trained for 50 Epochs.
### Model Architecture:

### Training Logs For Cifar10:

### Results:

## Vision Transformers(ViT's):

## Code Block Explanation:

### Block

### Embeddings

### MLP

### Attention

### Encoder

## Refernce Link:

[Spatial Transformers](https://towardsdatascience.com/review-stn-spatial-transformer-network-image-classification-d3cbd98a70aa)

## Contributors:
