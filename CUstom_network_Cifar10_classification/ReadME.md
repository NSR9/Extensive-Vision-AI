# Building a Custom network for CIFAR 10 classification
## Problem statement:-

1. Write a custom ResNet architecture for CIFAR10 that has the following architecture:
  1. PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]
  2. Layer1 -
    1. X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
    2. R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] 
    3. Add(X, R1)
  3. Layer 2 -
    1. Conv 3x3 [256k]
    2. MaxPooling2D
    3. BN
    4. ReLU
  4. Layer 3 -
    1. X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
    2. R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
    3. Add(X, R2)
  5. MaxPooling with Kernel Size 4
  6. FC Layer 
  7. SoftMax
2. Uses One Cycle Policy such that:
  1. Total Epochs = 24
  2. Max at Epoch = 5
  3. LRMIN = FIND
  4. LRMAX = FIND
  5. NO Annihilation
3. Uses this transform -RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8)
4. Batch size = 512
5. Target Accuracy: 90% (93% for late submission or double scores). 
6. NO score if your code is not modular. Your collab must be importing your GitHub package, and then just running the model. I should be able to find the custom_resnet.py model in your GitHub repo that you'd be training.

## Parameters:
- Batch Size - 512
- Transforms -
  - Padding(4,4)
  - Random Crop (32,32)
  - Flip Lr
  - Cutout(8,8)
- Model Total Parameters - 6,573,120
- Range Test
  - Max Lr - 0.02
  - Min Lr - 0.001
  - Epochs - 24
- Loss function - CrossEntropyLoss()
- Optimiser - SGD
  - Weight Decay - 0.05
  - Mometum - 0.9
- Scheduler - One Cycl2 Lr
  - epoch - 24
  - Max Lr - 0.012400000000000001
  - no of steps - 98
  - pct start - 0.0125
  - cyclic momentum -False

## Model Summary:

![image](https://user-images.githubusercontent.com/51078583/125988132-e0a327d5-8d04-4adc-b29f-c17b983c9ed1.png)

## Training logs:

      0%|          | 0/98 [00:00<?, ?it/s]EPOCH: 1 LR: 0.0012400000000000002
    Loss=1.4518868923187256 Batch_id=97 Accuracy=35.08: 100%|██████████| 98/98 [00:24<00:00,  3.98it/s]
      0%|          | 0/98 [00:00<?, ?it/s]
    Test set: Average loss: 0.0028, Accuracy: 4955/10000 (49.55%)

    EPOCH: 2 LR: 0.004031538194515345
    Loss=1.1477816104888916 Batch_id=97 Accuracy=52.64: 100%|██████████| 98/98 [00:24<00:00,  3.98it/s]
      0%|          | 0/98 [00:00<?, ?it/s]
    Test set: Average loss: 0.0024, Accuracy: 5907/10000 (59.07%)

    EPOCH: 3 LR: 0.0068230763890306904
    Loss=1.0303641557693481 Batch_id=97 Accuracy=60.69: 100%|██████████| 98/98 [00:24<00:00,  3.99it/s]
      0%|          | 0/98 [00:00<?, ?it/s]
    Test set: Average loss: 0.0017, Accuracy: 7063/10000 (70.63%)

    EPOCH: 4 LR: 0.009614614583546037
    Loss=0.9828583002090454 Batch_id=97 Accuracy=66.64: 100%|██████████| 98/98 [00:24<00:00,  3.99it/s]
      0%|          | 0/98 [00:00<?, ?it/s]
    Test set: Average loss: 0.0016, Accuracy: 7289/10000 (72.89%)

    EPOCH: 5 LR: 0.012398769630301102
    Loss=0.8007153272628784 Batch_id=97 Accuracy=70.50: 100%|██████████| 98/98 [00:24<00:00,  3.98it/s]
      0%|          | 0/98 [00:00<?, ?it/s]
    Test set: Average loss: 0.0013, Accuracy: 7832/10000 (78.32%)

    EPOCH: 6 LR: 0.011840546340985376
    Loss=0.8236434459686279 Batch_id=97 Accuracy=75.26: 100%|██████████| 98/98 [00:24<00:00,  3.99it/s]
      0%|          | 0/98 [00:00<?, ?it/s]
    Test set: Average loss: 0.0014, Accuracy: 7785/10000 (77.85%)

    EPOCH: 7 LR: 0.011282323051669648
    Loss=0.5569260120391846 Batch_id=97 Accuracy=77.75: 100%|██████████| 98/98 [00:24<00:00,  3.98it/s]
      0%|          | 0/98 [00:00<?, ?it/s]
    Test set: Average loss: 0.0012, Accuracy: 7967/10000 (79.67%)

    EPOCH: 8 LR: 0.010724099762353922
    Loss=0.6647427082061768 Batch_id=97 Accuracy=79.72: 100%|██████████| 98/98 [00:24<00:00,  3.97it/s]
      0%|          | 0/98 [00:00<?, ?it/s]
    Test set: Average loss: 0.0010, Accuracy: 8256/10000 (82.56%)

    EPOCH: 9 LR: 0.010165876473038196
    Loss=0.693509578704834 Batch_id=97 Accuracy=81.81: 100%|██████████| 98/98 [00:24<00:00,  3.99it/s]
      0%|          | 0/98 [00:00<?, ?it/s]
    Test set: Average loss: 0.0010, Accuracy: 8257/10000 (82.57%)

    EPOCH: 10 LR: 0.00960765318372247
    Loss=0.4754926264286041 Batch_id=97 Accuracy=83.45: 100%|██████████| 98/98 [00:24<00:00,  3.98it/s]
      0%|          | 0/98 [00:00<?, ?it/s]
    Test set: Average loss: 0.0010, Accuracy: 8399/10000 (83.99%)

    EPOCH: 11 LR: 0.009049429894406744
    Loss=0.4912489354610443 Batch_id=97 Accuracy=84.37: 100%|██████████| 98/98 [00:24<00:00,  3.97it/s]
      0%|          | 0/98 [00:00<?, ?it/s]
    Test set: Average loss: 0.0010, Accuracy: 8269/10000 (82.69%)

    EPOCH: 12 LR: 0.008491206605091017
    Loss=0.45821261405944824 Batch_id=97 Accuracy=85.16: 100%|██████████| 98/98 [00:24<00:00,  3.98it/s]
      0%|          | 0/98 [00:00<?, ?it/s]
    Test set: Average loss: 0.0008, Accuracy: 8638/10000 (86.38%)

    EPOCH: 13 LR: 0.00793298331577529
    Loss=0.3785838186740875 Batch_id=97 Accuracy=86.07: 100%|██████████| 98/98 [00:24<00:00,  3.98it/s]
      0%|          | 0/98 [00:00<?, ?it/s]
    Test set: Average loss: 0.0009, Accuracy: 8502/10000 (85.02%)

    EPOCH: 14 LR: 0.007374760026459565
    Loss=0.33560025691986084 Batch_id=97 Accuracy=87.26: 100%|██████████| 98/98 [00:24<00:00,  3.97it/s]
      0%|          | 0/98 [00:00<?, ?it/s]
    Test set: Average loss: 0.0008, Accuracy: 8737/10000 (87.37%)

    EPOCH: 15 LR: 0.006816536737143838
    Loss=0.2948436439037323 Batch_id=97 Accuracy=88.12: 100%|██████████| 98/98 [00:24<00:00,  3.98it/s]
      0%|          | 0/98 [00:00<?, ?it/s]
    Test set: Average loss: 0.0007, Accuracy: 8888/10000 (88.88%)

    EPOCH: 16 LR: 0.006258313447828112
    Loss=0.3658577799797058 Batch_id=97 Accuracy=88.79: 100%|██████████| 98/98 [00:24<00:00,  3.98it/s]
      0%|          | 0/98 [00:00<?, ?it/s]
    Test set: Average loss: 0.0007, Accuracy: 8878/10000 (88.78%)

    EPOCH: 17 LR: 0.005700090158512386
    Loss=0.29768887162208557 Batch_id=97 Accuracy=89.31: 100%|██████████| 98/98 [00:24<00:00,  3.97it/s]
      0%|          | 0/98 [00:00<?, ?it/s]
    Test set: Average loss: 0.0007, Accuracy: 8752/10000 (87.52%)

    EPOCH: 18 LR: 0.005141866869196659
    Loss=0.2803500294685364 Batch_id=97 Accuracy=89.78: 100%|██████████| 98/98 [00:24<00:00,  3.98it/s]
      0%|          | 0/98 [00:00<?, ?it/s]
    Test set: Average loss: 0.0007, Accuracy: 8850/10000 (88.50%)

    EPOCH: 19 LR: 0.004583643579880933
    Loss=0.25049862265586853 Batch_id=97 Accuracy=90.61: 100%|██████████| 98/98 [00:24<00:00,  3.99it/s]
      0%|          | 0/98 [00:00<?, ?it/s]
    Test set: Average loss: 0.0006, Accuracy: 8975/10000 (89.75%)

    EPOCH: 20 LR: 0.004025420290565206
    Loss=0.23058249056339264 Batch_id=97 Accuracy=91.35: 100%|██████████| 98/98 [00:24<00:00,  3.98it/s]
      0%|          | 0/98 [00:00<?, ?it/s]
    Test set: Average loss: 0.0006, Accuracy: 8994/10000 (89.94%)

    EPOCH: 21 LR: 0.0034671970012494797
    Loss=0.2315424084663391 Batch_id=97 Accuracy=91.90: 100%|██████████| 98/98 [00:24<00:00,  3.97it/s]
      0%|          | 0/98 [00:00<?, ?it/s]
    Test set: Average loss: 0.0006, Accuracy: 8988/10000 (89.88%)

    EPOCH: 22 LR: 0.002908973711933752
    Loss=0.23337122797966003 Batch_id=97 Accuracy=92.23: 100%|██████████| 98/98 [00:24<00:00,  3.97it/s]
      0%|          | 0/98 [00:00<?, ?it/s]
    Test set: Average loss: 0.0006, Accuracy: 9070/10000 (90.70%)

    EPOCH: 23 LR: 0.002350750422618026
    Loss=0.21399202942848206 Batch_id=97 Accuracy=92.99: 100%|██████████| 98/98 [00:24<00:00,  3.97it/s]
      0%|          | 0/98 [00:00<?, ?it/s]
    Test set: Average loss: 0.0005, Accuracy: 9120/10000 (91.20%)

    EPOCH: 24 LR: 0.0017925271333023001
    Loss=0.20310457050800323 Batch_id=97 Accuracy=93.56: 100%|██████████| 98/98 [00:24<00:00,  3.96it/s]

    Test set: Average loss: 0.0005, Accuracy: 9116/10000 (91.16%)
    
   
  
## LR Finder:
  
### One Cycle policy
Similar to Cyclic Learning Rate, but here we have only one Cycle. The correct combination of momemtum, weight decay, Learning rate, batch size does magic. One Cycle Policy will not increase accuracy, but the reasons to use it are

It reduces the time it takes to reach "near" to your accuracy.
It allows us to know if we are going right early on.
It let us know what kind of accuracies we can target with given model.
It reduces the cost of training.
It reduces the time to deploy
Both Cyclic Learning rate and One Cycle Policy was introduced by LESLIE SMITH

## Max LR:

LR Finder curve:

![image](https://user-images.githubusercontent.com/51078583/125988431-1e3ef71b-80b1-4e6c-9d29-54fc07fa6852.png)

The flatest part of the curve represents the max LR for One cycle LR. 

LR Finder Training logs:

    epoch = 1 Lr = 0.001  Loss=1.550249695777893 Batch_id=97 Accuracy=32.54: 100%|██████████| 98/98 [00:24<00:00,  4.07it/s]
    epoch = 2 Lr = 0.0029  Loss=1.4849601984024048 Batch_id=97 Accuracy=36.71: 100%|██████████| 98/98 [00:24<00:00,  3.92it/s]
    epoch = 3 Lr = 0.0048  Loss=1.377687692642212 Batch_id=97 Accuracy=35.53: 100%|██████████| 98/98 [00:25<00:00,  3.89it/s]
    epoch = 4 Lr = 0.006699999999999999  Loss=1.6627634763717651 Batch_id=97 Accuracy=27.13: 100%|██████████| 98/98 [00:24<00:00,  4.02it/s]
    epoch = 5 Lr = 0.0086  Loss=1.7587109804153442 Batch_id=97 Accuracy=22.70: 100%|██████████| 98/98 [00:24<00:00,  4.03it/s]
    epoch = 6 Lr = 0.0105  Loss=1.7244470119476318 Batch_id=97 Accuracy=22.54: 100%|██████████| 98/98 [00:24<00:00,  3.99it/s]
    epoch = 7 Lr = 0.012400000000000001  Loss=1.8090248107910156 Batch_id=97 Accuracy=20.00: 100%|██████████| 98/98 [00:24<00:00,  3.98it/s]
    epoch = 8 Lr = 0.014300000000000002  Loss=2.057267427444458 Batch_id=97 Accuracy=15.00: 100%|██████████| 98/98 [00:24<00:00,  4.01it/s]
    epoch = 9 Lr = 0.016200000000000003  Loss=1.976651668548584 Batch_id=97 Accuracy=15.97: 100%|██████████| 98/98 [00:24<00:00,  4.02it/s]
    epoch = 10 Lr = 0.0181  Loss=2.180305004119873 Batch_id=97 Accuracy=13.03: 100%|██████████| 98/98 [00:24<00:00,  4.01it/s]
    
**Max LR for oncecycle policy is at epoch 7 with 0.012400000000000001**
    
Lr for 24 epoch:

    [0.0012400000000000002,
     0.004031538194515345,
     0.0068230763890306904,
     0.009614614583546037,
     0.012398769630301102,
     0.011840546340985376,
     0.011282323051669648,
     0.010724099762353922,
     0.010165876473038196,
     0.00960765318372247,
     0.009049429894406744,
     0.008491206605091017,
     0.00793298331577529,
     0.007374760026459565,
     0.006816536737143838,
     0.006258313447828112,
     0.005700090158512386,
     0.005141866869196659,
     0.004583643579880933,
     0.004025420290565206,
     0.0034671970012494797,
     0.002908973711933752,
     0.002350750422618026,
     0.0017925271333023001]
  ![image](https://user-images.githubusercontent.com/51078583/126200689-aa5e37c1-6895-48fc-85d3-2c61095788ac.png)


**Max LR is at the 5th epoch**
  
## Results:

- Best Train Accuracy - 93.56%(24th epoch)
- Best Test Accuracy - 91.20%(23rd epoch)


### Validation loss curve:
![image](https://user-images.githubusercontent.com/51078583/126200573-8d6d3adf-206b-4046-8749-5261cfc32d38.png)

### Missclassified Images:

![image](https://user-images.githubusercontent.com/51078583/125989243-3ba98b23-c9dd-4d6d-adc1-4e2e600ac978.png)

### Accuracy for each class:

    Accuracy of plane : 85 %
    Accuracy of   car : 100 %
    Accuracy of  bird : 100 %
    Accuracy of   cat : 58 %
    Accuracy of  deer : 100 %
    Accuracy of   dog : 66 %
    Accuracy of  frog : 81 %
    Accuracy of horse : 100 %
    Accuracy of  ship : 100 %
    Accuracy of truck : 100 %

## Contributors:    
1. Avinash Ravi
2. Nandam Sriranga Chaitanya
3. Saroj Raj Das
4. Ujjwal Gupta
