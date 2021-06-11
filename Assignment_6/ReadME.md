# **Session 6 - (Late Assignement but on time)**
## Problem Statemnt

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

# Normalizartion

In image processing, normalization is a process that changes the range of pixel intensity values. 

The normalize is quite simple, it looks for the maximum intensity pixel (we will use a grayscale example here) and a minimum intensity and then will determine a factor that scales the min intensity to black and the max intensity to white. This is applied to every pixel in the image which produces the final result. 

There are mainly three types of Normalization techniques we will be discussing:-
* Batch Normalization 
* Layer Normalization 
* Group Normalization

## Batch Normalization 
## Layer Normalization
## Group Normalization

## Models and their Perfromance:
Dropout = 0.03

|Normalization|L1 Regularization|	L2 Regularization |  Best Train Accuracy	|Best Test Accuracy| Link to Logs|
|--|--|--|--|--|--|
|Layer Normalization| - | | | |
|Group Normalization| - | | | |
|Batch Normalization| Yes | | | |
|Layer Normalization| Yes | | | |
|Group Normalization| Yes | | | |
|Batch Normalization| Yes | Yes | | |

## Graphs and Plots (All 6 models mentioned above is compared)
|Graph 1: Training Loss for all 3 models together(Late Assignment Part)|Graph 2: Test/Validation Loss for all 3 models together|
|--|--|
|![image](https://user-images.githubusercontent.com/51078583/121725761-4bb08000-cb07-11eb-98de-296e91f6a74b.png)|![image](https://user-images.githubusercontent.com/51078583/121725803-59fe9c00-cb07-11eb-818f-ca5cb510792d.png)|

|Graph 3: Training Accuracy for all 3 models together(Late Assignment Part)|Graph 4: Test/Validation Accuracy for all 3 models together|
|--|--|
|![image](https://user-images.githubusercontent.com/51078583/121725858-6b47a880-cb07-11eb-8e3a-241b8395cbfc.png)|![image](https://user-images.githubusercontent.com/51078583/121725872-726eb680-cb07-11eb-8d88-ac7bf339ff76.png)|

##

