

# Assignment 3

We have written our neural network to solve the problem for the assignment. We have used the MNIST dataset and modified with adding the random no.

## **Data Representation**

The Data taken from MNIST data which contains images of digits in a 28X28 pixel representation  and the label column with the actual number associated with that image. We passed the loaded dataset into a Custom dataset Class, which is used to design our structured dataset. 

The class contains three functions namely __init__(), __getitem__() and __len__(). The purpose of each self explanatory, __init__() - for initializing the dataset taken, __getitem__() to get the next element of the dataset and __len__() for the length. The resultant dataset coming out from the class is a 4 element output namely 

The image tensor with shape {batch size}X1X28X28. 
* The random number one hot encoded tensor batchsizeX10
* The label for the MNIST image 
* The label for the sum value i.e. the random number + label of MNIST data. 

The Random number taken is converted into a one Hot encoded vector for the given reason: -
* To make it compatible for concatenating to the tensor output of the MNIST data.
* Keeping one large value in a tensor can manipulate the entire tensor weights in one direction. To keep it balanced hot encoding is preferred.

![Data Representation](https://user-images.githubusercontent.com/33301597/119178687-abc58080-ba8b-11eb-99f1-47d45adcdc2f.jpg)







## **Network Design**

* Network:
  * 7 conv layers
  * 2 max pool layers
  * 2 fully connected layers

* Reason for 7 conv layers 
  * kernel size 3
  * increased output channels in multiple of 2
  * 7th conv layers will give 10 outputs so that concatenation with encoded random tensor will of same dimemsion
  * Adjusted padding so that 7th conv layer be give 10 outputs

* Concatination of the two inputs:
  * The output of the 7th conv layer contains the 10 tensor values of the MNIST image input

![Network Architecture](https://user-images.githubusercontent.com/50147394/119181866-7bbdb380-ba72-11eb-9f8d-8f0e5718380a.jpg)

## **Network Summary**

![Network Summary](https://user-images.githubusercontent.com/50147394/119182925-ae1be080-ba73-11eb-9117-076d2cd8157c.jpg)

## **Loss Calculation for the best model**

If mnist accuracy falls below 95 then mnist loss will be used

* Other sum loss is used
* Experimented with couple of scenarios
  * using mnist_loss
  * using sum_loss
  * mnist_loss + sum_loss
* Using loss combination of above scenarios gave better results while comparing the result with individual ones

## **Training Logs for the best model**

![Training Logs](https://user-images.githubusercontent.com/50147394/119184501-bc6afc00-ba75-11eb-9716-91e350e4d5a4.JPG)


## **Testing Logs for the best model**

![Testing Logs](https://user-images.githubusercontent.com/50147394/119184617-e45a5f80-ba75-11eb-844c-6368ac093215.JPG)


## **Future upgrades** 

* Randomize the intake of loss(MNIST_loss / Sum_loss/ Sum_loss + MNIST_loss) for the backward propagation to improve the training of the the model for a larger scenario and avoiding overfitting for large epochs.


## **Team Members**

* Avinash Ravi
* Ujjwal Gupta
* Saroj Raj Das
* Nandam Sriranga Chaitanya


