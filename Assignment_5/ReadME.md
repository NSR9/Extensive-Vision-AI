## **Problem Statement:-**

## **Step Models:-**
### **Target, Result and Analysis of Step models:-**

|[Notebook_1(Github_Link)](https://github.com/NSR9/Extensive-Vision-AI/blob/main/Assignment_5/NoteBook_1_Low_params_5.0KParams.ipynb)![image](https://user-images.githubusercontent.com/51078583/120813331-69ac3c80-c56b-11eb-8295-81a26d3ffbdd.png)|[Notebook_2(Github_Link)](https://github.com/NSR9/Extensive-Vision-AI/blob/main/Assignment_5/NoteBook_2_Dropout&BN_8KParams.ipynb)![image](https://user-images.githubusercontent.com/51078583/120812326-8136f580-c56a-11eb-89b1-5e62f6c932cd.png)|
|--|--|

|[Notebook_3(Github_Link)](https://github.com/NSR9/Extensive-Vision-AI/blob/main/Assignment_5/Notebook_3_Image_augumentation_8.4Kparams.ipynb)![image](https://user-images.githubusercontent.com/51078583/120812436-97dd4c80-c56a-11eb-8679-5580c2b32b7c.png)|[Notebook_4(Github_Link)](https://github.com/NSR9/Extensive-Vision-AI/blob/main/Assignment_5/Notebook_4(Final)_StepLR_7.5KParam.ipynb)![image](https://user-images.githubusercontent.com/51078583/120815601-9bbe9e00-c56d-11eb-9494-9247046026c6.png)|
|--|--|

### **Feature Implementation in Step models:-**

**Here is the Tabular representation of the features implemented in the four step models**

| Model | Params Count | 1x1 conv layer| Maxpooling | Batch Normalization | Dropout | FC Layer| GAP | Image Augumentation | Optimizer | Schedular | 
|--|--|--|--|--|--|--|--|--|--|--|
|Notebook-1| 5,024 | Yes | Yes | No | No | No | No | No | SGD | No | 
|Notebook-2| 8,052 | Yes | Yes | Yes | Yes(0.1)| No | Yes | No | SGD | No | 
|Notebook-3| 8,444 | Yes | Yes | Yes | Yes(0.1)| No | Yes | Yes(RandomAffine, Color Jitter) | SGD | Yes(OneCycle) | 
|Notebook-4| 7,228  | Yes | Yes | Yes | Yes(0.05) | No | Yes | Yes(RandomRoatation) | SGD | Yes(StepLR) | 


**Note:-**
* ReLU, Batch Normalization and Dropout if Implemented is added to each Conv Layer expect the prediction Layer.
* Image Augumentation was applied on the traning dataset while the test dataset was left untouched.

### **Receptive Field Calulation of Models:-**
**Formulae**

<img src="https://user-images.githubusercontent.com/51078583/120814031-17b7e680-c56c-11eb-8a87-7bd01dd2c849.png" width=400 height=400>

#### Notebook-1
[Notebook_1(Github_Link)](https://github.com/NSR9/Extensive-Vision-AI/blob/main/Assignment_5/NoteBook_1_Low_params_5.0KParams.ipynb)
|Layer|In_Dim|Out_Dim|In_channels|Out_channels|Pad|Stride|Jin|Jout|Rf_in|Rf_out|
|--|--|--|--|--|--|--|--|--|--|--|
#### Notebook-2
[Notebook_2(Github_Link)](https://github.com/NSR9/Extensive-Vision-AI/blob/main/Assignment_5/NoteBook_2_Dropout&BN_8KParams.ipynb)
|Layer|In_Dim|Out_Dim|In_channels|Out_channels|Pad|Stride|Jin|Jout|Rf_in|Rf_out|
|--|--|--|--|--|--|--|--|--|--|--|
#### Notebook-3
[Notebook_3(Github_Link)](https://github.com/NSR9/Extensive-Vision-AI/blob/main/Assignment_5/Notebook_3_Image_augumentation_8.4Kparams.ipynb)
|Layer|In_Dim|Out_Dim|In_channels|Out_channels|Pad|Stride|Jin|Jout|Rf_in|Rf_out|
|--|--|--|--|--|--|--|--|--|--|--|
#### Notebook-4
[Notebook_4(Github_Link)](https://github.com/NSR9/Extensive-Vision-AI/blob/main/Assignment_5/Notebook_4(Final)_StepLR_7.5KParam.ipynb)
|Layer|In_Dim|Out_Dim|In_channels|Out_channels|Pad|Stride|Jin|Jout|Rf_in|Rf_out|
|--|--|--|--|--|--|--|--|--|--|--|

## **Proposed Network (Best Network - Notebook_4):-**

### **Network Block:-**
#### Conv Block 1
* 2D Convolution number of kernels 8, followed with Batch Normalization and 2D Dropout of 0.05
* 2D Convolution number of kernels 8, followed with Batch Normalization and 2D Dropout of 0.05
#### Transition Layer 1
* 2D Max Pooling to reduce the size of the channel to 12
#### Conv Block 2
* 2D Convolution number of kernels 10, followed with Batch Normalization and 2D Dropout of 0.05
* 2D Convolution number of kernels 10, followed with Batch Normalization and 2D Dropout of 0.05
* 2D Convolution number of kernels 10, followed with Batch Normalization and 2D Dropout of 0.05
* 2D Convolution number of kernels 16, followed with Batch Normalization and 2D Dropout of 0.1
* 2D Convolution number of kernels 16, followed with Batch Normalization and 2D Dropout of 0.1
#### Global Average Pooling
* Global Average pooling with a size 6 and Padding 1 to return a 16 x 1 x 1 as output dimensions
#### Conv Block 3
* 2D Convolution number of kernels 10, followed with Batch Normalization and 2D Dropout of 0.1

## Model Summary:-
![image](https://user-images.githubusercontent.com/51078583/120816521-72ead880-c56e-11eb-9c11-d0b1682fff2d.png)

### Goals Achived:-
In the Notebook 3 we achieved the goal of 99.4% accuracy. But the model was not stable with that accuracy. Notebook_4 achieved all the required goals. 
* The Target was achieved with **less than 8,000 Parameters**, exactly **7,228**.
* Achieved Accuracy of **99.40% in the 7th Epoch** itself and the model was consistant with an accuracy greated than the same throughout the 15 Epochs. 
* Highest achieved accuracy was of **99.47 at the 11th Epoch**. 
* The model was **not overfitting** and the Gap beween the Training and testing accuracy was very less(Can be seen in the Training-Validation curve.)

### Logs of Final Model:-
```
Epoch1 : Loss=0.18589811027050018  Accuracy=83.81 Batch_id=468: 100%|██████████| 469/469 [00:15<00:00, 30.89it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0872, Accuracy: 9812/10000 (98.12%)
Epoch2 : Loss=0.05694916471838951  Accuracy=97.15 Batch_id=468: 100%|██████████| 469/469 [00:15<00:00, 30.33it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0611, Accuracy: 9839/10000 (98.39%)
Epoch3 : Loss=0.1879713386297226  Accuracy=97.75 Batch_id=468: 100%|██████████| 469/469 [00:15<00:00, 30.67it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0364, Accuracy: 9902/10000 (99.02%)
Epoch4 : Loss=0.1543048620223999  Accuracy=98.03 Batch_id=468: 100%|██████████| 469/469 [00:15<00:00, 30.46it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0335, Accuracy: 9907/10000 (99.07%)
Epoch5 : Loss=0.04255908727645874  Accuracy=98.21 Batch_id=468: 100%|██████████| 469/469 [00:15<00:00, 30.08it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0298, Accuracy: 9918/10000 (99.18%)
Epoch6 : Loss=0.04420612379908562  Accuracy=98.38 Batch_id=468: 100%|██████████| 469/469 [00:15<00:00, 30.41it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0274, Accuracy: 9927/10000 (99.27%)
Epoch7 : Loss=0.014818566851317883  Accuracy=98.53 Batch_id=468: 100%|██████████| 469/469 [00:15<00:00, 30.40it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0212, Accuracy: 9940/10000 (99.40%)
Epoch8 : Loss=0.04427675902843475  Accuracy=98.68 Batch_id=468: 100%|██████████| 469/469 [00:15<00:00, 29.85it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0206, Accuracy: 9942/10000 (99.42%)
Epoch9 : Loss=0.061607468873262405  Accuracy=98.67 Batch_id=468: 100%|██████████| 469/469 [00:15<00:00, 30.77it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0196, Accuracy: 9942/10000 (99.42%)
Epoch10 : Loss=0.04594561830163002  Accuracy=98.69 Batch_id=468: 100%|██████████| 469/469 [00:15<00:00, 30.69it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0193, Accuracy: 9943/10000 (99.43%)
Epoch11 : Loss=0.013693585060536861  Accuracy=98.72 Batch_id=468: 100%|██████████| 469/469 [00:15<00:00, 30.68it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0190, Accuracy: 9947/10000 (99.47%)
Epoch12 : Loss=0.026162752881646156  Accuracy=98.72 Batch_id=468: 100%|██████████| 469/469 [00:15<00:00, 30.50it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0193, Accuracy: 9943/10000 (99.43%)
Epoch13 : Loss=0.03216176852583885  Accuracy=98.75 Batch_id=468: 100%|██████████| 469/469 [00:15<00:00, 30.94it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0189, Accuracy: 9944/10000 (99.44%)
Epoch14 : Loss=0.032220084220170975  Accuracy=98.81 Batch_id=468: 100%|██████████| 469/469 [00:15<00:00, 30.92it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0186, Accuracy: 9941/10000 (99.41%)
Epoch15 : Loss=0.039210837334394455  Accuracy=98.80 Batch_id=468: 100%|██████████| 469/469 [00:15<00:00, 31.20it/s]
Test set: Average loss: 0.0186, Accuracy: 9941/10000 (99.41%)
```
## Statistics (Best Model):-

### Confusion Matrix:-
![image](https://user-images.githubusercontent.com/51078583/120818041-dde8df00-c56f-11eb-8e16-b181f35581e1.png)

### Training-Validation Curve:-
![image](https://user-images.githubusercontent.com/51078583/120818119-f22cdc00-c56f-11eb-9f05-094773989f82.png)

### Incorrect image:-
Some of the incorrect predicted images.

![image](https://user-images.githubusercontent.com/51078583/120818292-18eb1280-c570-11eb-9cf0-1d49092c7b74.png)

## Refernces:-
* [Pytorch Trasnforms Documentation](https://github.com/NSR9/Extensive-Vision-AI/blob/main/Assignment_5/Notebook_4(Final)_StepLR_7.5KParam.ipynb)
## Contributors:-

1. Avinash Ravi
2. Nandam Sriranga Chaitanya
3. Saroj Raj Das
4. Ujjwal Gupta
