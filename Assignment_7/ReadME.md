# Assignemnt 7 (Earl)
## Problem statement:-
1. change the code such that it uses GPU
    * change the architecture to C1C2C3C40  (No MaxPooling, but 3 3x3 layers with stride of 2 instead) (If you can figure out how to use Dilated kernels here instead of MP or          strided convolution, then 200pts extra!)
    * total RF must be more than 44
    * one of the layers must use Depthwise Separable Convolution
    * one of the layers must use Dilated Convolution
    * use GAP (compulsory):- add FC after GAP to target #of classes (optional)
    * use albumentation library and apply:
        * horizontal flip
        * shiftScaleRotate
        * coarseDropout (max_holes = 1, max_height=16px, max_width=1, min_holes = 1, min_height=16px, min_width=16px, fill_value=(mean of your dataset), mask_fill_value = None)
            achieve 85% accuracy, as many epochs as you want. Total Params to be less than 200k. 
         * grayscale
    * achieve 85% accuracy, as many epochs as you want. Total Params to be less than 100k. 

## WorkFlow:-
### Enchancements used:-

* We have not used maxpooling a any time. 
* For the Transition block 1 only 1x1 conv s used to reduce the number of params.
* For the Transition block 2 we have used **Dilation Convolution** to decrease the size of image and get a higher receptive field output
* For the Transition block 3 we have used normal convolutions with stride as 2. 
* We have used **6 Depth wise Seperable convolutions** . 
* Use Cross Entropy for 

## Data Augumentation:-

The data augumentation techniques used are:-
* HorizontalFlip
* ShiftScaleRotate
* CoarseDropout
* Normalize
* ToGray
* ToTensorV2

```
class Transforms:
    def __init__(self, transforms: A.Compose):
        self.transforms = transforms

    def __call__(self, img, *args, **kwargs):
        return self.transforms(image=np.array(img))

train_transform = A.Compose([
            A.HorizontalFlip(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101, always_apply=False, p=0.5),
            A.CoarseDropout(max_holes = 1, max_height=16, max_width=16, min_holes = 1, min_height=16, min_width=16, fill_value=(0.4914, 0.4822, 0.4465), mask_fill_value = None),
            A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
            A.ToGray(),
            ToTensorV2(),
        ])

test_transform = A.Compose([
    A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
    ToTensorV2(),
])

train_transform = Transforms(train_transform)
test_transform = Transforms(test_transform)
```
## Best Model:-

### Model Summary:-

### Training Logs:-
        0%|          | 0/391 [00:00<?, ?it/s]Epoch 1:
      Loss=1.9617748260498047 Batch_id=390 Accuracy=18.60: 100%|██████████| 391/391 [00:16<00:00, 23.21it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0155, Accuracy: 11099/50000 (22.20%)

      Epoch 2:
      Loss=1.8240715265274048 Batch_id=390 Accuracy=28.88: 100%|██████████| 391/391 [00:16<00:00, 23.73it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0136, Accuracy: 17150/50000 (34.30%)

      Epoch 3:
      Loss=1.6940200328826904 Batch_id=390 Accuracy=34.10: 100%|██████████| 391/391 [00:16<00:00, 24.30it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0122, Accuracy: 20488/50000 (40.98%)

      Epoch 4:
      Loss=1.5552923679351807 Batch_id=390 Accuracy=37.93: 100%|██████████| 391/391 [00:16<00:00, 24.25it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0116, Accuracy: 22092/50000 (44.18%)

      Epoch 5:
      Loss=1.642897605895996 Batch_id=390 Accuracy=41.19: 100%|██████████| 391/391 [00:16<00:00, 24.14it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0107, Accuracy: 24544/50000 (49.09%)

      Epoch 6:
      Loss=1.509184718132019 Batch_id=390 Accuracy=43.58: 100%|██████████| 391/391 [00:16<00:00, 24.20it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0101, Accuracy: 25986/50000 (51.97%)

      Epoch 7:
      Loss=1.4026966094970703 Batch_id=390 Accuracy=46.18: 100%|██████████| 391/391 [00:16<00:00, 24.28it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0097, Accuracy: 27268/50000 (54.54%)

      Epoch 8:
      Loss=1.5222337245941162 Batch_id=390 Accuracy=47.90: 100%|██████████| 391/391 [00:16<00:00, 24.19it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0099, Accuracy: 27095/50000 (54.19%)

      Epoch 9:
      Loss=1.1760590076446533 Batch_id=390 Accuracy=48.98: 100%|██████████| 391/391 [00:16<00:00, 24.11it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0088, Accuracy: 29388/50000 (58.78%)

      Epoch 10:
      Loss=1.2728935480117798 Batch_id=390 Accuracy=50.41: 100%|██████████| 391/391 [00:16<00:00, 24.27it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0087, Accuracy: 29796/50000 (59.59%)

      Epoch 11:
      Loss=1.292845368385315 Batch_id=390 Accuracy=51.59: 100%|██████████| 391/391 [00:16<00:00, 24.31it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0089, Accuracy: 29774/50000 (59.55%)

      Epoch 12:
      Loss=1.2440195083618164 Batch_id=390 Accuracy=52.93: 100%|██████████| 391/391 [00:16<00:00, 24.08it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0080, Accuracy: 31464/50000 (62.93%)

      Epoch 13:
      Loss=1.4440639019012451 Batch_id=390 Accuracy=53.98: 100%|██████████| 391/391 [00:16<00:00, 24.05it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0078, Accuracy: 32028/50000 (64.06%)

      Epoch 14:
      Loss=1.4682139158248901 Batch_id=390 Accuracy=54.83: 100%|██████████| 391/391 [00:16<00:00, 24.09it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0078, Accuracy: 32036/50000 (64.07%)

      Epoch 15:
      Loss=1.4051693677902222 Batch_id=390 Accuracy=55.80: 100%|██████████| 391/391 [00:16<00:00, 24.03it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0074, Accuracy: 32941/50000 (65.88%)

      Epoch 16:
      Loss=1.3370463848114014 Batch_id=390 Accuracy=56.74: 100%|██████████| 391/391 [00:16<00:00, 23.99it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0072, Accuracy: 33581/50000 (67.16%)

      Epoch 17:
      Loss=1.0382769107818604 Batch_id=390 Accuracy=57.55: 100%|██████████| 391/391 [00:16<00:00, 24.00it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0071, Accuracy: 33713/50000 (67.43%)

      Epoch 18:
      Loss=1.0502631664276123 Batch_id=390 Accuracy=58.08: 100%|██████████| 391/391 [00:16<00:00, 23.88it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0072, Accuracy: 33880/50000 (67.76%)

      Epoch 19:
      Loss=1.0337384939193726 Batch_id=390 Accuracy=59.22: 100%|██████████| 391/391 [00:16<00:00, 24.14it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0073, Accuracy: 33787/50000 (67.57%)

      Epoch 20:
      Loss=1.1833324432373047 Batch_id=390 Accuracy=59.65: 100%|██████████| 391/391 [00:16<00:00, 24.03it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0065, Accuracy: 35394/50000 (70.79%)

      Epoch 21:
      Loss=1.0973570346832275 Batch_id=390 Accuracy=60.32: 100%|██████████| 391/391 [00:16<00:00, 23.96it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0064, Accuracy: 35225/50000 (70.45%)

      Epoch 22:
      Loss=1.1229403018951416 Batch_id=390 Accuracy=60.71: 100%|██████████| 391/391 [00:16<00:00, 24.03it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0067, Accuracy: 35078/50000 (70.16%)

      Epoch 23:
      Loss=1.022588849067688 Batch_id=390 Accuracy=61.74: 100%|██████████| 391/391 [00:16<00:00, 23.91it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0061, Accuracy: 36221/50000 (72.44%)

      Epoch 24:
      Loss=1.0384981632232666 Batch_id=390 Accuracy=62.38: 100%|██████████| 391/391 [00:16<00:00, 24.25it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0061, Accuracy: 36080/50000 (72.16%)

      Epoch 25:
      Loss=1.012056589126587 Batch_id=390 Accuracy=62.89: 100%|██████████| 391/391 [00:16<00:00, 23.85it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0059, Accuracy: 36488/50000 (72.98%)

      Epoch 26:
      Loss=1.214317798614502 Batch_id=390 Accuracy=62.97: 100%|██████████| 391/391 [00:16<00:00, 24.11it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0060, Accuracy: 36395/50000 (72.79%)

      Epoch 27:
      Loss=0.9769196510314941 Batch_id=390 Accuracy=63.44: 100%|██████████| 391/391 [00:16<00:00, 24.28it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0062, Accuracy: 36236/50000 (72.47%)

      Epoch 28:
      Loss=1.067617416381836 Batch_id=390 Accuracy=64.13: 100%|██████████| 391/391 [00:16<00:00, 23.98it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0057, Accuracy: 37129/50000 (74.26%)

      Epoch 29:
      Loss=1.0719388723373413 Batch_id=390 Accuracy=64.51: 100%|██████████| 391/391 [00:16<00:00, 23.81it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0057, Accuracy: 37277/50000 (74.55%)

      Epoch 30:
      Loss=1.0482361316680908 Batch_id=390 Accuracy=64.72: 100%|██████████| 391/391 [00:16<00:00, 24.13it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0058, Accuracy: 37019/50000 (74.04%)

      Epoch 31:
      Loss=0.8320412635803223 Batch_id=390 Accuracy=65.20: 100%|██████████| 391/391 [00:16<00:00, 23.88it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0055, Accuracy: 37599/50000 (75.20%)

      Epoch 32:
      Loss=0.8156550526618958 Batch_id=390 Accuracy=65.19: 100%|██████████| 391/391 [00:16<00:00, 23.98it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0050, Accuracy: 38794/50000 (77.59%)

      Epoch 33:
      Loss=0.9869725108146667 Batch_id=390 Accuracy=65.69: 100%|██████████| 391/391 [00:16<00:00, 23.89it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0051, Accuracy: 38497/50000 (76.99%)

      Epoch 34:
      Loss=0.7811394929885864 Batch_id=390 Accuracy=66.04: 100%|██████████| 391/391 [00:16<00:00, 24.08it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0051, Accuracy: 38473/50000 (76.95%)

      Epoch 35:
      Loss=0.7895585894584656 Batch_id=390 Accuracy=66.30: 100%|██████████| 391/391 [00:16<00:00, 23.92it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0048, Accuracy: 39075/50000 (78.15%)

      Epoch 36:
      Loss=1.0878249406814575 Batch_id=390 Accuracy=66.68: 100%|██████████| 391/391 [00:16<00:00, 23.86it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0049, Accuracy: 39113/50000 (78.23%)

      Epoch 37:
      Loss=0.754012942314148 Batch_id=390 Accuracy=67.01: 100%|██████████| 391/391 [00:16<00:00, 24.19it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0050, Accuracy: 38931/50000 (77.86%)

      Epoch 38:
      Loss=1.1218807697296143 Batch_id=390 Accuracy=67.05: 100%|██████████| 391/391 [00:16<00:00, 23.73it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0049, Accuracy: 39138/50000 (78.28%)

      Epoch 39:
      Loss=0.7922338247299194 Batch_id=390 Accuracy=67.44: 100%|██████████| 391/391 [00:16<00:00, 23.69it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0048, Accuracy: 39336/50000 (78.67%)

      Epoch 40:
      Loss=0.867035984992981 Batch_id=390 Accuracy=67.84: 100%|██████████| 391/391 [00:16<00:00, 23.95it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0047, Accuracy: 39343/50000 (78.69%)

      Epoch 41:
      Loss=1.1128127574920654 Batch_id=390 Accuracy=67.98: 100%|██████████| 391/391 [00:16<00:00, 23.93it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0048, Accuracy: 39313/50000 (78.63%)

      Epoch 42:
      Loss=0.8835676312446594 Batch_id=390 Accuracy=67.97: 100%|██████████| 391/391 [00:16<00:00, 23.90it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0047, Accuracy: 39642/50000 (79.28%)

      Epoch 43:
      Loss=0.887199878692627 Batch_id=390 Accuracy=68.21: 100%|██████████| 391/391 [00:16<00:00, 23.81it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0047, Accuracy: 39634/50000 (79.27%)

      Epoch 44:
      Loss=0.9129770994186401 Batch_id=390 Accuracy=67.97: 100%|██████████| 391/391 [00:16<00:00, 24.00it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0045, Accuracy: 39953/50000 (79.91%)

      Epoch 45:
      Loss=0.879075825214386 Batch_id=390 Accuracy=68.47: 100%|██████████| 391/391 [00:16<00:00, 23.88it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0046, Accuracy: 39759/50000 (79.52%)

      Epoch 46:
      Loss=0.8798263669013977 Batch_id=390 Accuracy=68.94: 100%|██████████| 391/391 [00:16<00:00, 23.60it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0046, Accuracy: 39757/50000 (79.51%)

      Epoch 47:
      Loss=0.9070954322814941 Batch_id=390 Accuracy=69.10: 100%|██████████| 391/391 [00:16<00:00, 23.55it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0045, Accuracy: 39749/50000 (79.50%)

      Epoch 48:
      Loss=0.9693851470947266 Batch_id=390 Accuracy=69.19: 100%|██████████| 391/391 [00:16<00:00, 23.27it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0045, Accuracy: 39987/50000 (79.97%)

      Epoch 49:
      Loss=0.8708375692367554 Batch_id=390 Accuracy=69.33: 100%|██████████| 391/391 [00:16<00:00, 23.53it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0045, Accuracy: 40178/50000 (80.36%)

      Epoch 50:
      Loss=1.1177469491958618 Batch_id=390 Accuracy=69.82: 100%|██████████| 391/391 [00:16<00:00, 23.49it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0042, Accuracy: 40749/50000 (81.50%)

      Epoch 51:
      Loss=0.6330642104148865 Batch_id=390 Accuracy=69.92: 100%|██████████| 391/391 [00:16<00:00, 23.85it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0042, Accuracy: 40645/50000 (81.29%)

      Epoch 52:
      Loss=0.7560447454452515 Batch_id=390 Accuracy=69.79: 100%|██████████| 391/391 [00:16<00:00, 23.47it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0042, Accuracy: 40669/50000 (81.34%)

      Epoch 53:
      Loss=0.8890754580497742 Batch_id=390 Accuracy=69.72: 100%|██████████| 391/391 [00:16<00:00, 23.79it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0041, Accuracy: 40877/50000 (81.75%)

      Epoch 54:
      Loss=1.0481340885162354 Batch_id=390 Accuracy=69.96: 100%|██████████| 391/391 [00:16<00:00, 23.94it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0041, Accuracy: 40835/50000 (81.67%)

      Epoch 55:
      Loss=0.9756792187690735 Batch_id=390 Accuracy=70.07: 100%|██████████| 391/391 [00:16<00:00, 23.67it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0042, Accuracy: 40578/50000 (81.16%)

      Epoch 56:
      Loss=1.1141549348831177 Batch_id=390 Accuracy=70.41: 100%|██████████| 391/391 [00:16<00:00, 23.79it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0041, Accuracy: 40823/50000 (81.65%)

      Epoch 57:
      Loss=0.763458251953125 Batch_id=390 Accuracy=70.49: 100%|██████████| 391/391 [00:16<00:00, 24.06it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0040, Accuracy: 41129/50000 (82.26%)

      Epoch 58:
      Loss=0.8565123677253723 Batch_id=390 Accuracy=70.27: 100%|██████████| 391/391 [00:16<00:00, 24.12it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0039, Accuracy: 41287/50000 (82.57%)

      Epoch 59:
      Loss=0.6991275548934937 Batch_id=390 Accuracy=70.80: 100%|██████████| 391/391 [00:16<00:00, 24.20it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0038, Accuracy: 41601/50000 (83.20%)

      Epoch 60:
      Loss=1.1233488321304321 Batch_id=390 Accuracy=70.94: 100%|██████████| 391/391 [00:16<00:00, 23.99it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0039, Accuracy: 41303/50000 (82.61%)

      Epoch 61:
      Loss=0.6903630495071411 Batch_id=390 Accuracy=71.33: 100%|██████████| 391/391 [00:16<00:00, 23.98it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0040, Accuracy: 41142/50000 (82.28%)

      Epoch 62:
      Loss=0.8220556378364563 Batch_id=390 Accuracy=71.47: 100%|██████████| 391/391 [00:16<00:00, 23.98it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0039, Accuracy: 41479/50000 (82.96%)

      Epoch 63:
      Loss=0.7586281299591064 Batch_id=390 Accuracy=71.52: 100%|██████████| 391/391 [00:16<00:00, 23.90it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0037, Accuracy: 41819/50000 (83.64%)

      Epoch 64:
      Loss=1.0481477975845337 Batch_id=390 Accuracy=71.44: 100%|██████████| 391/391 [00:16<00:00, 24.00it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0037, Accuracy: 41932/50000 (83.86%)

      Epoch 65:
      Loss=0.8157222867012024 Batch_id=390 Accuracy=71.89: 100%|██████████| 391/391 [00:16<00:00, 23.86it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0037, Accuracy: 41884/50000 (83.77%)

      Epoch 66:
      Loss=0.7052044868469238 Batch_id=390 Accuracy=72.15: 100%|██████████| 391/391 [00:16<00:00, 23.86it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0036, Accuracy: 42079/50000 (84.16%)

      Epoch 67:
      Loss=0.7799423933029175 Batch_id=390 Accuracy=71.64: 100%|██████████| 391/391 [00:16<00:00, 23.84it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0037, Accuracy: 41947/50000 (83.89%)

      Epoch 68:
      Loss=0.8780487179756165 Batch_id=390 Accuracy=72.03: 100%|██████████| 391/391 [00:16<00:00, 23.63it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0036, Accuracy: 42018/50000 (84.04%)

      Epoch 69:
      Loss=0.6862533688545227 Batch_id=390 Accuracy=72.53: 100%|██████████| 391/391 [00:16<00:00, 23.81it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0035, Accuracy: 42383/50000 (84.77%)

      Epoch 70:
      Loss=0.8790459632873535 Batch_id=390 Accuracy=72.49: 100%|██████████| 391/391 [00:16<00:00, 23.77it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0035, Accuracy: 42170/50000 (84.34%)

      Epoch 71:
      Loss=0.7624297142028809 Batch_id=390 Accuracy=72.52: 100%|██████████| 391/391 [00:16<00:00, 23.91it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0035, Accuracy: 42308/50000 (84.62%)

      Epoch 72:
      Loss=0.827390193939209 Batch_id=390 Accuracy=72.96: 100%|██████████| 391/391 [00:16<00:00, 23.95it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0034, Accuracy: 42574/50000 (85.15%)

      Epoch 73:
      Loss=0.8999192118644714 Batch_id=390 Accuracy=72.92: 100%|██████████| 391/391 [00:16<00:00, 23.78it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0035, Accuracy: 42245/50000 (84.49%)

      Epoch 74:
      Loss=0.7837992906570435 Batch_id=390 Accuracy=72.94: 100%|██████████| 391/391 [00:16<00:00, 23.98it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0034, Accuracy: 42555/50000 (85.11%)

      Epoch 75:
      Loss=0.84306401014328 Batch_id=390 Accuracy=73.16: 100%|██████████| 391/391 [00:16<00:00, 23.70it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0033, Accuracy: 42601/50000 (85.20%)

      Epoch 76:
      Loss=0.6454547643661499 Batch_id=390 Accuracy=73.31: 100%|██████████| 391/391 [00:16<00:00, 24.01it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0033, Accuracy: 42755/50000 (85.51%)

      Epoch 77:
      Loss=0.7639096975326538 Batch_id=390 Accuracy=73.51: 100%|██████████| 391/391 [00:16<00:00, 23.76it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0033, Accuracy: 42694/50000 (85.39%)

      Epoch 78:
      Loss=0.8150988817214966 Batch_id=390 Accuracy=74.03: 100%|██████████| 391/391 [00:16<00:00, 23.94it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0033, Accuracy: 42787/50000 (85.57%)

      Epoch 79:
      Loss=0.7749282717704773 Batch_id=390 Accuracy=73.74: 100%|██████████| 391/391 [00:16<00:00, 23.89it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0032, Accuracy: 42901/50000 (85.80%)

      Epoch 80:
      Loss=0.8054191470146179 Batch_id=390 Accuracy=74.18: 100%|██████████| 391/391 [00:16<00:00, 23.62it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0032, Accuracy: 42933/50000 (85.87%)

      Epoch 81:
      Loss=0.7883496284484863 Batch_id=390 Accuracy=74.22: 100%|██████████| 391/391 [00:16<00:00, 23.76it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0032, Accuracy: 42899/50000 (85.80%)

      Epoch 82:
      Loss=0.7530765533447266 Batch_id=390 Accuracy=73.91: 100%|██████████| 391/391 [00:16<00:00, 23.67it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0032, Accuracy: 43028/50000 (86.06%)

      Epoch 83:
      Loss=0.6697109341621399 Batch_id=390 Accuracy=74.22: 100%|██████████| 391/391 [00:16<00:00, 23.04it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0032, Accuracy: 43051/50000 (86.10%)

      Epoch 84:
      Loss=0.6346662640571594 Batch_id=390 Accuracy=74.24: 100%|██████████| 391/391 [00:16<00:00, 23.15it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0032, Accuracy: 43078/50000 (86.16%)

      Epoch 85:
      Loss=0.7280442714691162 Batch_id=390 Accuracy=74.39: 100%|██████████| 391/391 [00:16<00:00, 23.58it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0031, Accuracy: 43149/50000 (86.30%)

      Epoch 86:
      Loss=0.8343936204910278 Batch_id=390 Accuracy=74.51: 100%|██████████| 391/391 [00:16<00:00, 23.59it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0031, Accuracy: 43169/50000 (86.34%)

      Epoch 87:
      Loss=0.7086585760116577 Batch_id=390 Accuracy=74.74: 100%|██████████| 391/391 [00:16<00:00, 23.89it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0031, Accuracy: 43167/50000 (86.33%)

      Epoch 88:
      Loss=0.8070831298828125 Batch_id=390 Accuracy=74.54: 100%|██████████| 391/391 [00:16<00:00, 23.34it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0031, Accuracy: 43179/50000 (86.36%)

      Epoch 89:
      Loss=0.8599007725715637 Batch_id=390 Accuracy=74.78: 100%|██████████| 391/391 [00:16<00:00, 23.48it/s]
        0%|          | 0/391 [00:00<?, ?it/s]
      Test set: Average loss: 0.0031, Accuracy: 43214/50000 (86.43%)

      Epoch 90:
      Loss=0.693084716796875 Batch_id=390 Accuracy=75.02: 100%|██████████| 391/391 [00:16<00:00, 23.62it/s]

      Test set: Average loss: 0.0031, Accuracy: 43237/50000 (86.47%)

### Goals Achieved:-
* Total Params - 
* Best Training Accuracy - 
* Best Testing Accuracy - 

### Accuracy of each class:-

## Contributors:-
1. Avinash Ravi
2. Nandam Sriranga Chaitanya
3. Saroj Raj Das
4. Ujjwal Gupta
