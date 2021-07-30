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

### 1. Localization Network:

It is a simple neural network with a few convolution layers and a few dense layers. It predicts the parameters of transformation as output. These parameters determine the angle by which the input has to be rotated, the amount of translation to be done, and the scaling factor required to focus on the region of interest in the input feature map.
 
### 2. Grid Generator:

The transformation parameters predicted by the localization net are used in the form of an affine transformation matrix of size 2 x 3 for each image in the batch. An affine transformation is one which preserves points, straight lines and planes. Parallel lines remain parallel after affine transformation. Rotation, scaling and translation are all affine transformations.

![image](https://user-images.githubusercontent.com/51078583/127388037-68615834-1b48-44a7-92fb-4abb266df9d8.png)

Here, (xti,yti) are the target coordinates of the target grid in the output feature map, (xsi,ysi) are the input coordinates in the input feature map, and Aθ is the affine transformation matrix. T is the transformation and A is the matrix representing the affine transformation. θ11, θ12, θ21, θ22 determine the angle by which the image has to be rotated. θ13, θ23 determine the translations along width and height of the image respectively. Thus we obtain a sampling grid of transformed indices.


### 3. Sampler:

This is the last part of the spatial transformer network. We have the input feature map and also the parameterized sampling grid with us now. To perform the sampling, we give the feature map U and sampling grid Tθ(G) as input to the sampler. The sampling kernel is applied to the source coordinates using the parameters θ and we get the output V.

![image](https://user-images.githubusercontent.com/51078583/127388329-bcc89a75-6558-439d-bd25-52ad90d72415.png)

## CIFAR10 with STN:

Implement that [Spatial Transformer Code](https://brsoff.github.io/tutorials/intermediate/spatial_transformer_tutorial.html) for CIFAR10 and submit. Must have proper readme and must be trained for 50 Epochs.
- [COLAB LINK TO THE CODE]()
- [GITHUB LINK TO THE CODE](https://github.com/NSR9/Extensive-Vision-AI/blob/main/Assignment_12/Spatial%20Transformation/Assignment12_Spatial_Transformer.ipynb)

### Training Logs For Cifar10:

```
Train Epoch: 1 [0/50000 (0%)]	Loss: 0.690737
Train Epoch: 1 [32000/50000 (64%)]	Loss: 0.637498
/usr/local/lib/python3.7/dist-packages/torch/nn/_reduction.py:42: UserWarning: size_average and reduce args will be deprecated, please use reduction='sum' instead.
  warnings.warn(warning.format(ret))
Test set: Average loss: 1.0878, Accuracy: 6460/10000 (65%)

Train Epoch: 2 [0/50000 (0%)]	Loss: 0.659649
Train Epoch: 2 [32000/50000 (64%)]	Loss: 0.605217

Test set: Average loss: 1.1406, Accuracy: 6348/10000 (63%)

Train Epoch: 3 [0/50000 (0%)]	Loss: 0.947379
Train Epoch: 3 [32000/50000 (64%)]	Loss: 0.609259

Test set: Average loss: 1.0727, Accuracy: 6476/10000 (65%)

Train Epoch: 4 [0/50000 (0%)]	Loss: 0.785464
Train Epoch: 4 [32000/50000 (64%)]	Loss: 1.027550

Test set: Average loss: 1.0692, Accuracy: 6560/10000 (66%)

Train Epoch: 5 [0/50000 (0%)]	Loss: 0.527283
Train Epoch: 5 [32000/50000 (64%)]	Loss: 0.894939

Test set: Average loss: 1.0406, Accuracy: 6590/10000 (66%)

Train Epoch: 6 [0/50000 (0%)]	Loss: 0.489604
Train Epoch: 6 [32000/50000 (64%)]	Loss: 0.670998

Test set: Average loss: 1.0595, Accuracy: 6571/10000 (66%)

Train Epoch: 7 [0/50000 (0%)]	Loss: 0.651692
Train Epoch: 7 [32000/50000 (64%)]	Loss: 0.609619

Test set: Average loss: 1.1275, Accuracy: 6344/10000 (63%)

Train Epoch: 8 [0/50000 (0%)]	Loss: 0.602985
Train Epoch: 8 [32000/50000 (64%)]	Loss: 0.512781

Test set: Average loss: 1.1054, Accuracy: 6427/10000 (64%)

Train Epoch: 9 [0/50000 (0%)]	Loss: 0.438184
Train Epoch: 9 [32000/50000 (64%)]	Loss: 0.616304

Test set: Average loss: 1.0655, Accuracy: 6549/10000 (65%)

Train Epoch: 10 [0/50000 (0%)]	Loss: 0.605649
Train Epoch: 10 [32000/50000 (64%)]	Loss: 0.750937

Test set: Average loss: 1.1548, Accuracy: 6306/10000 (63%)

Train Epoch: 11 [0/50000 (0%)]	Loss: 0.777266
Train Epoch: 11 [32000/50000 (64%)]	Loss: 0.554822

Test set: Average loss: 1.0937, Accuracy: 6459/10000 (65%)

Train Epoch: 12 [0/50000 (0%)]	Loss: 0.635354
Train Epoch: 12 [32000/50000 (64%)]	Loss: 0.509758

Test set: Average loss: 1.1027, Accuracy: 6460/10000 (65%)

Train Epoch: 13 [0/50000 (0%)]	Loss: 0.562828
Train Epoch: 13 [32000/50000 (64%)]	Loss: 0.449412

Test set: Average loss: 1.1076, Accuracy: 6484/10000 (65%)

Train Epoch: 14 [0/50000 (0%)]	Loss: 0.521123
Train Epoch: 14 [32000/50000 (64%)]	Loss: 0.494318

Test set: Average loss: 1.1371, Accuracy: 6431/10000 (64%)

Train Epoch: 15 [0/50000 (0%)]	Loss: 0.712082
Train Epoch: 15 [32000/50000 (64%)]	Loss: 0.597805

Test set: Average loss: 1.1397, Accuracy: 6350/10000 (64%)

Train Epoch: 16 [0/50000 (0%)]	Loss: 0.668810
Train Epoch: 16 [32000/50000 (64%)]	Loss: 0.665830

Test set: Average loss: 1.1190, Accuracy: 6479/10000 (65%)

Train Epoch: 17 [0/50000 (0%)]	Loss: 0.549515
Train Epoch: 17 [32000/50000 (64%)]	Loss: 0.701603

Test set: Average loss: 1.1024, Accuracy: 6603/10000 (66%)

Train Epoch: 18 [0/50000 (0%)]	Loss: 0.495903
Train Epoch: 18 [32000/50000 (64%)]	Loss: 0.343320

Test set: Average loss: 1.0944, Accuracy: 6575/10000 (66%)

Train Epoch: 19 [0/50000 (0%)]	Loss: 0.402828
Train Epoch: 19 [32000/50000 (64%)]	Loss: 0.373760

Test set: Average loss: 1.1413, Accuracy: 6420/10000 (64%)

Train Epoch: 20 [0/50000 (0%)]	Loss: 0.501965
Train Epoch: 20 [32000/50000 (64%)]	Loss: 0.535749

Test set: Average loss: 1.1169, Accuracy: 6501/10000 (65%)

Train Epoch: 21 [0/50000 (0%)]	Loss: 0.365870
Train Epoch: 21 [32000/50000 (64%)]	Loss: 0.409410

Test set: Average loss: 1.1215, Accuracy: 6580/10000 (66%)

Train Epoch: 22 [0/50000 (0%)]	Loss: 0.404387
Train Epoch: 22 [32000/50000 (64%)]	Loss: 0.394960

Test set: Average loss: 1.1201, Accuracy: 6494/10000 (65%)

Train Epoch: 23 [0/50000 (0%)]	Loss: 0.528931
Train Epoch: 23 [32000/50000 (64%)]	Loss: 0.385287

Test set: Average loss: 1.1408, Accuracy: 6448/10000 (64%)

Train Epoch: 24 [0/50000 (0%)]	Loss: 0.646410
Train Epoch: 24 [32000/50000 (64%)]	Loss: 0.395826

Test set: Average loss: 1.1543, Accuracy: 6426/10000 (64%)

Train Epoch: 25 [0/50000 (0%)]	Loss: 0.561092
Train Epoch: 25 [32000/50000 (64%)]	Loss: 0.397083

Test set: Average loss: 1.2451, Accuracy: 6281/10000 (63%)

Train Epoch: 26 [0/50000 (0%)]	Loss: 0.495252
Train Epoch: 26 [32000/50000 (64%)]	Loss: 0.659562

Test set: Average loss: 1.1696, Accuracy: 6399/10000 (64%)

Train Epoch: 27 [0/50000 (0%)]	Loss: 0.625578
Train Epoch: 27 [32000/50000 (64%)]	Loss: 0.647164

Test set: Average loss: 1.1949, Accuracy: 6375/10000 (64%)

Train Epoch: 28 [0/50000 (0%)]	Loss: 0.614327
Train Epoch: 28 [32000/50000 (64%)]	Loss: 0.412665

Test set: Average loss: 1.2323, Accuracy: 6316/10000 (63%)

Train Epoch: 29 [0/50000 (0%)]	Loss: 0.543714
Train Epoch: 29 [32000/50000 (64%)]	Loss: 0.340727

Test set: Average loss: 1.2430, Accuracy: 6226/10000 (62%)

Train Epoch: 30 [0/50000 (0%)]	Loss: 0.892907
Train Epoch: 30 [32000/50000 (64%)]	Loss: 0.430613

Test set: Average loss: 1.1597, Accuracy: 6521/10000 (65%)

Train Epoch: 31 [0/50000 (0%)]	Loss: 0.415773
Train Epoch: 31 [32000/50000 (64%)]	Loss: 0.434429

Test set: Average loss: 1.1901, Accuracy: 6418/10000 (64%)

Train Epoch: 32 [0/50000 (0%)]	Loss: 0.342234
Train Epoch: 32 [32000/50000 (64%)]	Loss: 0.534007

Test set: Average loss: 1.3508, Accuracy: 6027/10000 (60%)

Train Epoch: 33 [0/50000 (0%)]	Loss: 0.715538
Train Epoch: 33 [32000/50000 (64%)]	Loss: 0.400395

Test set: Average loss: 1.1956, Accuracy: 6468/10000 (65%)

Train Epoch: 34 [0/50000 (0%)]	Loss: 0.429562
Train Epoch: 34 [32000/50000 (64%)]	Loss: 0.569019

Test set: Average loss: 1.2090, Accuracy: 6453/10000 (65%)

Train Epoch: 35 [0/50000 (0%)]	Loss: 0.366523
Train Epoch: 35 [32000/50000 (64%)]	Loss: 0.489472

Test set: Average loss: 1.3332, Accuracy: 6189/10000 (62%)

Train Epoch: 36 [0/50000 (0%)]	Loss: 0.485867
Train Epoch: 36 [32000/50000 (64%)]	Loss: 0.413372

Test set: Average loss: 1.1821, Accuracy: 6462/10000 (65%)

Train Epoch: 37 [0/50000 (0%)]	Loss: 0.283405
Train Epoch: 37 [32000/50000 (64%)]	Loss: 0.654417

Test set: Average loss: 1.2022, Accuracy: 6461/10000 (65%)

Train Epoch: 38 [0/50000 (0%)]	Loss: 0.463182
Train Epoch: 38 [32000/50000 (64%)]	Loss: 0.356603

Test set: Average loss: 1.2599, Accuracy: 6162/10000 (62%)

Train Epoch: 39 [0/50000 (0%)]	Loss: 0.412473
Train Epoch: 39 [32000/50000 (64%)]	Loss: 0.306318

Test set: Average loss: 1.2179, Accuracy: 6437/10000 (64%)

Train Epoch: 40 [0/50000 (0%)]	Loss: 0.314837
Train Epoch: 40 [32000/50000 (64%)]	Loss: 0.581722

Test set: Average loss: 1.2299, Accuracy: 6375/10000 (64%)

Train Epoch: 41 [0/50000 (0%)]	Loss: 0.369637
Train Epoch: 41 [32000/50000 (64%)]	Loss: 0.195445

Test set: Average loss: 1.3251, Accuracy: 6261/10000 (63%)

Train Epoch: 42 [0/50000 (0%)]	Loss: 0.510895
Train Epoch: 42 [32000/50000 (64%)]	Loss: 0.221283

Test set: Average loss: 1.1952, Accuracy: 6526/10000 (65%)

Train Epoch: 43 [0/50000 (0%)]	Loss: 0.307904
Train Epoch: 43 [32000/50000 (64%)]	Loss: 0.468787

Test set: Average loss: 1.2352, Accuracy: 6456/10000 (65%)

Train Epoch: 44 [0/50000 (0%)]	Loss: 0.489498
Train Epoch: 44 [32000/50000 (64%)]	Loss: 0.324359

Test set: Average loss: 1.1934, Accuracy: 6504/10000 (65%)

Train Epoch: 45 [0/50000 (0%)]	Loss: 0.385749
Train Epoch: 45 [32000/50000 (64%)]	Loss: 0.334431

Test set: Average loss: 1.2019, Accuracy: 6467/10000 (65%)

Train Epoch: 46 [0/50000 (0%)]	Loss: 0.340109
Train Epoch: 46 [32000/50000 (64%)]	Loss: 0.474386

Test set: Average loss: 1.2887, Accuracy: 6286/10000 (63%)

Train Epoch: 47 [0/50000 (0%)]	Loss: 0.315563
Train Epoch: 47 [32000/50000 (64%)]	Loss: 0.302742

Test set: Average loss: 1.2732, Accuracy: 6392/10000 (64%)

Train Epoch: 48 [0/50000 (0%)]	Loss: 0.413236
Train Epoch: 48 [32000/50000 (64%)]	Loss: 0.313085

Test set: Average loss: 1.5666, Accuracy: 5934/10000 (59%)

Train Epoch: 49 [0/50000 (0%)]	Loss: 0.847494
Train Epoch: 49 [32000/50000 (64%)]	Loss: 0.294018

Test set: Average loss: 1.2313, Accuracy: 6398/10000 (64%)

Train Epoch: 50 [0/50000 (0%)]	Loss: 0.284545
Train Epoch: 50 [32000/50000 (64%)]	Loss: 0.475773

Test set: Average loss: 1.2774, Accuracy: 6241/10000 (62%)
```

### Results:
- Epoches - 50
- Best Testing Accuracy - 65.26%(EPOCH 42)

#### 	Visualizatio of STN:

![image](https://user-images.githubusercontent.com/51078583/127677554-3be8ada3-1da5-4acb-bb83-fb4a099f1a3a.png)


## Refernce Link:

- [Spatial Transformers](https://towardsdatascience.com/review-stn-spatial-transformer-network-image-classification-d3cbd98a70aa)

## Contributors:
1. Avinash Ravi
2. Nandam Sriranga Chaitanya
3. Saroj Raj Das
4. Ujjwal Gupta

