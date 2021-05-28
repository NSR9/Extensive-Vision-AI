## Problem Statement

### WRITE DOWN THE CODE FOR MNIST CLASSIFICATION WITH FOLLOWING CONSTRAINTS:-
* 99.4% validation accuracy
* Less than 20k Parameters
* You can use anything from above you want. 
* Less than 20 Epochs
* Have used BN, Dropout, a Fully connected layer, have used GAP. 

## Proposed Network (Best Network):-

### Network Block :

#### Conv Block 1
* 2D Convolution number of kernels 8, followed with Batch Normalization and 2D Dropout of 0.1
* 2D Convolution number of kernels 16, followed with Batch Normalization and 2D Dropout of 0.1
* 2D Convolution number of kernels 32, followed with Batch Normalization and 2D Dropout of 0.1

#### Transition Layer 1
* 2D Max Pooling to reduce the size of the channel to 14
* 2d Convolution with kernel size 1 reducing the number of channels to 8

#### Conv Block 2
* 2D Convolution number of kernels 12, followed with Batch Normalization and 2D Dropout of 0.1
* 2D Convolution number of kernels 24, followed with Batch Normalization and 2D Dropout of 0.1
* 2D Convolution number of kernels 32, followed with Batch Normalization and 2D Dropout of 0.1

#### Transition Layer 2
* 2D Max Pooling to reduce the size of the channel to 7
* 2d Convolution with kernel size 1 reducing the number of channels to 8

#### Conv Block 3
* 2D Convolution number of kernels 16, followed with Batch Normalization and 2D Dropout of 0.1
* 2D Convolution number of kernels 10 (Avoid Batch Normalization and Dropout in Last layer before GAP)

#### Global Average Pooling
* Global Average pooling with a size 3 and no Padding to return a 10 1 1 as the value to go to log_softmax

## Model Summary:-

### Expirement Model:-

### Best Model Summary:-

#### Enchancements to the Model:-

* Activation Function as ReLU is used after conv layers
* MaxPool Layer of 2 x 2 is used twice in the network.
* Conv 1 x 1 is used in the transition layer for reducing the number of channels
* Added batch normalization after every conv layer
* Added dropout of 0.1 after each conv layer
* Added Global average pooling to get output classes.
* Use learning rate of 0.01 and momentum 0.9

* **Paramerters Used** - **19,750** 
* **Best Accuracy** - **99.44% at the 16th Epoch**

![image](https://user-images.githubusercontent.com/51078583/119997847-c9479c80-bfed-11eb-9028-a3edd9892116.png)

#### Logs for Best Model:-

Test set: Average loss: 0.0749, Accuracy: 9765/10000 (97.65%)

HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))

Test set: Average loss: 0.0400, Accuracy: 9877/10000 (98.77%)

HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))

Test set: Average loss: 0.0350, Accuracy: 9882/10000 (98.82%)

HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))

Test set: Average loss: 0.0300, Accuracy: 9902/10000 (99.02%)

HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))

Test set: Average loss: 0.0299, Accuracy: 9898/10000 (98.98%)

HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))

Test set: Average loss: 0.0273, Accuracy: 9899/10000 (98.99%)

HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))

Test set: Average loss: 0.0235, Accuracy: 9920/10000 (99.20%)

HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))

Test set: Average loss: 0.0214, Accuracy: 9934/10000 (99.34%)

HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))

Test set: Average loss: 0.0221, Accuracy: 9928/10000 (99.28%)

HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))

Test set: Average loss: 0.0222, Accuracy: 9922/10000 (99.22%)

HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))

Test set: Average loss: 0.0218, Accuracy: 9927/10000 (99.27%)

HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))

Test set: Average loss: 0.0206, Accuracy: 9935/10000 (99.35%)

HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))

Test set: Average loss: 0.0219, Accuracy: 9927/10000 (99.27%)

HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))

Test set: Average loss: 0.0222, Accuracy: 9925/10000 (99.25%)

HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))

Test set: Average loss: 0.0205, Accuracy: 9933/10000 (99.33%)

HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))

Test set: Average loss: 0.0179, Accuracy: 9944/10000 (99.44%)

HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))

Test set: Average loss: 0.0191, Accuracy: 9939/10000 (99.39%)

HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))

Test set: Average loss: 0.0194, Accuracy: 9929/10000 (99.29%)

HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))

Test set: Average loss: 0.0188, Accuracy: 9940/10000 (99.40%)


