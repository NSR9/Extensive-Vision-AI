---

---

# Assignment 3

We have written our neural network to solve the problem for the assignment. We have used the MNIST dataset and modified with adding the random no.

## Data Representation

The Data taken from MNIST data which contains images of digits in a 28X28 pixel representation  and the label column with the actual number associated with that image. We passed the loaded dataset into a Custom dataset Class, which is used to design our structured dataset. 

The class contains three functions namely __init__(), __getitem__() and __len__(). The purpose of each self explanatory, __init__() - for initializing the dataset taken, __getitem__() to get the next element of the dataset and __len__() for the length. The resultant dataset coming out from the class is a 4 element output namely 

The image tensor with shape {batch size}X1X28X28, 
* The random number one hot encoded tensor batchsizeX10
* The label for the MNIST image 
* The label for the sum value i.e. the random number + label of MNIST data. 

The Random number taken is converted into a one Hot encoded vector for the given reason: -
* To make it compatible for concatenating to the tensor output of the MNIST data.
* Keeping one large value in a tensor can manipulate the entire tensor weights in one direction. To keep it balanced hot encoding is preferred.

![Data Representation](https://user-images.githubusercontent.com/33301597/119178687-abc58080-ba8b-11eb-99f1-47d45adcdc2f.jpg)




## The network 

1. Our network has 7 convolution layers and 2 max pooling layers before merging them to the 2 fully connected layers.

![Network Architecture](https://user-images.githubusercontent.com/50147394/119181866-7bbdb380-ba72-11eb-9f8d-8f0e5718380a.jpg)
