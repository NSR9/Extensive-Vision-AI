## **Problem Statement**

### **WRITE DOWN THE CODE FOR MNIST CLASSIFICATION WITH FOLLOWING CONSTRAINTS:-**
* 99.4% validation accuracy
* Less than 20k Parameters
* Less than 20 Epochs
* Can use BN, Dropout, a Fully connected layer, have used GAP.

## **Proposed Network (Best Network):-**

### **Network Block :**

![image](https://user-images.githubusercontent.com/51078583/120019024-8f829000-c005-11eb-8e6d-2756b71a4f72.png)

#### Conv Block 1
* 2D Convolution number of kernels 8, followed with Batch Normalization and 2D Dropout of 0.1 
* 2D Convolution number of kernels 16, followed with Batch Normalization and 2D Dropout of 0.1
* 2D Convolution number of kernels 32, followed with Batch Normalization and 2D Dropout of 0.1
#### Transition Layer 1
* 2D Max Pooling to reduce the size of the channel to 14
* 2d Convolution with kernel size 1 reducing the number of channels to 8
#### Conv Block 2
* 2D Convolution number of kernels 16, followed with Batch Normalization and 2D Dropout of 0.1 
* 2D Convolution number of kernels 16, followed with Batch Normalization and 2D Dropout of 0.1
* 2D Convolution number of kernels 16, followed with Batch Normalization and 2D Dropout of 0.1
#### Transition Layer 2
* 2D Max Pooling to reduce the size of the channel to 7
#### Conv Block 3
* 2D Convolution number of kernels 16, followed with Batch Normalization and 2D Dropout of 0.1 
* 2D Convolution number of kernels 10 (Avoid Batch Normalization and Dropout in Last layer before GAP)
#### Global Average Pooling
* Global Average pooling with a size 3 and no Padding to return a 10 x 1 x 1 as the value to go to log_softmax 

## **Best Model Summary:-**
#### [Github_link](https://github.com/NSR9/Extensive-Vision-AI/blob/main/Assignment_4/Part%202/Final_Submission_Assignment_4_NB_3.ipynb)
### Enhancements to the Model:
       * Random Rotation of 5 applied as Data Augumentation Methodlogy
       * Activation Function as ReLU is used after conv layers
       * MaxPool Layer of 2 x 2 is used twice in the network. 
       * Conv 1X1 is used in the transition layer for reducing the number of channels
       * Added batch normalization after every conv layer
       * Added dropout of 0.1 after each conv layer
       * Added Global average pooling to get output classes.
       * Use learning rate of 0.01 and momentum 0.9
       
### Goals Achieved:-
* Parameters count reduced to as low as **15,970**
* Achieved an highest accuracy of **99.50%** at the **18th Epoch**. Getting an accuracy of greater than **99.40%** for the first time at **12th epoch** itself. 
* Achieved a final Accuracy of **99.47%** after 19 Epochs.

![image](https://user-images.githubusercontent.com/51078583/120013438-598ddd80-bffe-11eb-9198-771222359f61.png)

### Logs for Best Model:-

         0%|          | 0/469 [00:00<?, ?it/s]/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:64: UserWarning: Implicit dimension choice for log_softmax has been deprecated. Change the call to include dim=X as an argument.
       epoch=1 loss=0.0804116353 batch_id=00468: 100%|██████████| 469/469 [00:17<00:00, 27.07it/s]
         0%|          | 0/469 [00:00<?, ?it/s]
       Test set: Average loss: 0.0564, Accuracy: 9812/10000 (98.12%)

       epoch=2 loss=0.0772533491 batch_id=00468: 100%|██████████| 469/469 [00:16<00:00, 27.59it/s]
         0%|          | 0/469 [00:00<?, ?it/s]
       Test set: Average loss: 0.0408, Accuracy: 9866/10000 (98.66%)

       epoch=3 loss=0.0261944626 batch_id=00468: 100%|██████████| 469/469 [00:16<00:00, 27.60it/s]
         0%|          | 0/469 [00:00<?, ?it/s]
       Test set: Average loss: 0.0329, Accuracy: 9899/10000 (98.99%)

       epoch=4 loss=0.1168910488 batch_id=00468: 100%|██████████| 469/469 [00:16<00:00, 28.11it/s]
         0%|          | 0/469 [00:00<?, ?it/s]
       Test set: Average loss: 0.0254, Accuracy: 9911/10000 (99.11%)

       epoch=5 loss=0.0491798073 batch_id=00468: 100%|██████████| 469/469 [00:17<00:00, 27.15it/s]
         0%|          | 0/469 [00:00<?, ?it/s]
       Test set: Average loss: 0.0242, Accuracy: 9922/10000 (99.22%)

       epoch=6 loss=0.0509031415 batch_id=00468: 100%|██████████| 469/469 [00:17<00:00, 27.43it/s]
         0%|          | 0/469 [00:00<?, ?it/s]
       Test set: Average loss: 0.0261, Accuracy: 9911/10000 (99.11%)

       epoch=7 loss=0.0247863010 batch_id=00468: 100%|██████████| 469/469 [00:18<00:00, 25.69it/s]
         0%|          | 0/469 [00:00<?, ?it/s]
       Test set: Average loss: 0.0221, Accuracy: 9921/10000 (99.21%)

       epoch=8 loss=0.0185720865 batch_id=00468: 100%|██████████| 469/469 [00:18<00:00, 25.52it/s]
         0%|          | 0/469 [00:00<?, ?it/s]
       Test set: Average loss: 0.0234, Accuracy: 9923/10000 (99.23%)

       epoch=9 loss=0.0328470059 batch_id=00468: 100%|██████████| 469/469 [00:17<00:00, 26.63it/s]
         0%|          | 0/469 [00:00<?, ?it/s]
       Test set: Average loss: 0.0212, Accuracy: 9934/10000 (99.34%)

       epoch=10 loss=0.0394887738 batch_id=00468: 100%|██████████| 469/469 [00:17<00:00, 26.30it/s]
         0%|          | 0/469 [00:00<?, ?it/s]
       Test set: Average loss: 0.0200, Accuracy: 9932/10000 (99.32%)

       epoch=11 loss=0.1504232436 batch_id=00468: 100%|██████████| 469/469 [00:17<00:00, 26.86it/s]
         0%|          | 0/469 [00:00<?, ?it/s]
       Test set: Average loss: 0.0203, Accuracy: 9933/10000 (99.33%)

       epoch=12 loss=0.0531375371 batch_id=00468: 100%|██████████| 469/469 [00:17<00:00, 26.66it/s]
         0%|          | 0/469 [00:00<?, ?it/s]
       Test set: Average loss: 0.0184, Accuracy: 9941/10000 (99.41%)

       epoch=13 loss=0.0361075923 batch_id=00468: 100%|██████████| 469/469 [00:17<00:00, 26.72it/s]
         0%|          | 0/469 [00:00<?, ?it/s]
       Test set: Average loss: 0.0178, Accuracy: 9935/10000 (99.35%)

       epoch=14 loss=0.0506523736 batch_id=00468: 100%|██████████| 469/469 [00:17<00:00, 26.96it/s]
         0%|          | 0/469 [00:00<?, ?it/s]
       Test set: Average loss: 0.0164, Accuracy: 9945/10000 (99.45%)

       epoch=15 loss=0.0154768182 batch_id=00468: 100%|██████████| 469/469 [00:17<00:00, 27.33it/s]
         0%|          | 0/469 [00:00<?, ?it/s]
       Test set: Average loss: 0.0180, Accuracy: 9946/10000 (99.46%)

       epoch=16 loss=0.0440198146 batch_id=00468: 100%|██████████| 469/469 [00:17<00:00, 26.95it/s]
         0%|          | 0/469 [00:00<?, ?it/s]
       Test set: Average loss: 0.0172, Accuracy: 9947/10000 (99.47%)

       epoch=17 loss=0.0216456298 batch_id=00468: 100%|██████████| 469/469 [00:17<00:00, 27.02it/s]
         0%|          | 0/469 [00:00<?, ?it/s]
       Test set: Average loss: 0.0200, Accuracy: 9935/10000 (99.35%)

       epoch=18 loss=0.1076599658 batch_id=00468: 100%|██████████| 469/469 [00:17<00:00, 27.02it/s]
         0%|          | 0/469 [00:00<?, ?it/s]
       Test set: Average loss: 0.0160, Accuracy: 9950/10000 (99.50%)

       epoch=19 loss=0.0563756712 batch_id=00468: 100%|██████████| 469/469 [00:17<00:00, 26.35it/s]
       Test set: Average loss: 0.0157, Accuracy: 9947/10000 (99.47%)
      
### **Validation Loss Curve:-**

![image](https://user-images.githubusercontent.com/51078583/120013747-c5704600-bffe-11eb-840e-ad2ae3d49969.png)


## **Expirement Models:-**

#### **Experiment Model 1 Summary:-**
#### [Github_link](https://github.com/NSR9/Extensive-Vision-AI/blob/main/Assignment_4/Part%202/Experiment_Nb_1.ipynb)
**Enhancements**

       * Activation Function as ReLU is used after conv layers
       * MaxPool Layer of 2 x 2 is used twice in the network.
       * Added batch normalization after every conv layer
       * Added dropout of 0.069 after each conv layer
       * Added Global average pooling before the FC layer and then added the FC to get output classes.
       * Use learning rate of 0.02 and momentum 0.8

* **Paramerters Used** - **14,906** 
* **Best Accuracy** - **99.49% at the 16th Epoch**

![image](https://user-images.githubusercontent.com/51078583/120001574-8daed180-bff1-11eb-90ae-291d5cfc5ed0.png)

**Drawbacks**
* Fully connected layers are used. 
* 1 x 1 Conv layers not used to reduce the number of channels 


### **Experiment Model 2 Summary:-**
#### [Github_link](https://github.com/NSR9/Extensive-Vision-AI/blob/main/Assignment_4/Part%202/Experiment_assignment_NB_2.ipynb)
**Enhancements**

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

**Drawbacks**
* Number of Parameters can be reduced more.
* The Achieved accuracy above 99.40 is not constant but fluctuating

## **Refernce Link:-**

[Kaggle]( https://www.kaggle.com/enwei26/mnist-digits-pytorch-cnn-99)

## **Contributors:-**

1. Avinash Ravi
2. Nandam Sriranga Chaitanya
3. Saroj Raj Das
4. Ujjwal Gupta
