# **Session 6 - (Late Assignment but on time)**
## Problem Statement:-

You are making 3 versions of your 5th assignment's best model (or pick one from best assignments):
1. Network with Group Normalization
2. Network with Layer Normalization
3. Network with L1 + BN
4. Network with Group Normalization + L1(Late Assignment Part)
5. Network with Layer Normalization + L2(Late Assignment Part)
6. Network with L1 + L2 + BN(Late Assignment Part)

Create these graphs:
* Graph 1: Training Loss for all 3 models together(Late Assignment Part)
* Graph 2: Test/Validation Loss for all 3 models together
* Graph 3: Training Accuracy for all 3 models together(Late Assignment Part)
* Graph 4: Test/Validation Accuracy for all 3 models together

Find 20 misclassified images for each of the 3 models, and show them as a 5x4 image matrix in 3 separately annotated images. 

# Normalization:-

In image processing, normalization is a process that changes the range of pixel intensity values. 

The normalize is quite simple, it looks for the maximum intensity pixel (we will use a grayscale example here) and a minimum intensity and then will determine a factor that scales the min intensity to black and the max intensity to white. This is applied to every pixel in the image which produces the final result. 

The Basic Formulae of implementaion of normalization can be represented in the following experession:-
![image](https://user-images.githubusercontent.com/51078583/121730596-86b5b200-cb0d-11eb-8d06-898729c46467.png)


There are mainly three types of Normalization techniques we will be discussing:-
* Batch Normalization 
* Layer Normalization 
* Group Normalization

![image](https://user-images.githubusercontent.com/51078583/121730799-c7adc680-cb0d-11eb-91bb-5bd2169169a4.png)

## Batch Normalization:-
It can be considered as the rescaling of image with respect to the channels. 

Mathematically, BN layer transforms each input in the current mini-batch by subtracting the input mean in the current mini-batch and dividing it by the standard deviation.
Below given is the Mathematical implication of the Batch Normalization. 


For example:-


## Layer Normalization:-
Layer normalization normalizes input across the features instead of normalizing input features across the batch dimension in batch normalization.

For example:-
## Group Normalization:-

As the name suggests, Group Normalization normalizes over group of channels for each training examples. We can say that, Group Norm is in between Instance Norm and Layer Norm.

When we put all the channels into a single group, group normalization becomes Layer normalization. And, when we put each channel into different groups it becomes Instance normalization.

For example:-

## Models and their Performance:-
Dropout = 0.03

Epoches = 20

|Normalization|L1 Regularization|	L2 Regularization | Params Count | Best Train Accuracy	|Best Test Accuracy| Link to Logs|
|--|--|--|--|--|--|--|
|Layer Normalization| - | - |43208 |98.91 |99.62|[Layer Norm Logs]() | 
|Group Normalization| - | - | 7704| 98.72|99.51 | |
|Batch Normalization| Yes | - |7704 |97.84 |99.35 | |
|Layer Normalization| Yes | - |43208 |97.33 |99.06 | |
|Group Normalization| Yes | - |7704| 98.26|99.34 | |
|Batch Normalization| Yes | Yes |7704 |97.87 | 99.4| |
 
## Graphs and Plots (All 6 models mentioned above is compared):-
|Graph 1: Training Loss for all 3 models together(Late Assignment Part)|Graph 2: Test/Validation Loss for all 3 models together|
|--|--|
|![image](https://user-images.githubusercontent.com/51078583/121725761-4bb08000-cb07-11eb-98de-296e91f6a74b.png)|![image](https://user-images.githubusercontent.com/51078583/121725803-59fe9c00-cb07-11eb-818f-ca5cb510792d.png)|

|Graph 3: Training Accuracy for all 3 models together(Late Assignment Part)|Graph 4: Test/Validation Accuracy for all 3 models together|
|--|--|
|![image](https://user-images.githubusercontent.com/51078583/121725858-6b47a880-cb07-11eb-8e3a-241b8395cbfc.png)|![image](https://user-images.githubusercontent.com/51078583/121725872-726eb680-cb07-11eb-8d88-ac7bf339ff76.png)|

## Misclassified Images:-

|Group Normalization|Layer Normalization|
|--|--|
|![image](https://user-images.githubusercontent.com/51078583/121727153-27ee3980-cb09-11eb-9172-063f4e97c418.png)|![image](https://user-images.githubusercontent.com/51078583/121726971-e9587f00-cb08-11eb-992a-6da138d4404a.png)|

|Layer Normalization + L1|Group Normalization + L1|
|--|--|
|![image](https://user-images.githubusercontent.com/51078583/121727381-76033d00-cb09-11eb-9229-dc66640ba2e1.png)|![image](https://user-images.githubusercontent.com/51078583/121727434-861b1c80-cb09-11eb-9bf8-eb6ffdd70f90.png)|

|Batch Normalization + L1|Batch Normalization + L1 + L2|
|--|--|
|![image](https://user-images.githubusercontent.com/51078583/121727909-2bce8b80-cb0a-11eb-9cc7-1a151565f973.png)|![image](https://user-images.githubusercontent.com/51078583/121727948-3db02e80-cb0a-11eb-9bab-6c2d4b1dba49.png)|
