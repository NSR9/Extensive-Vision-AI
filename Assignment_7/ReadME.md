# Assignemnt 7
## Problem statement:-

Fix the given network with below conditions:

1. change the code such that it uses GPU
2. change the architecture to C1C2C3C40 (No MaxPooling, but 3 3x3 layers with stride of 2 instead) (If you can figure out how to use Dilated kernels here instead of MP or strided convolution, then 200pts extra!)
3. total RF must be more than 52
4. two of the layers must use Depthwise Separable Convolution
5. one of the layers must use Dilated Convolution
6. use GAP (compulsory mapped to # of classes):- CANNOT add FC after GAP to target #of classes 
7. use albumentation library and apply:
   horizontal flip
   shiftScaleRotate 
   coarseDropout (max_holes = 1, max_height=16px, max_width=1, min_holes = 1, min_height=16px, min_width=16px, fill_value=(mean of your dataset), mask_fill_value = None)
   grayscale
8. achieve 87% accuracy, as many epochs as you want. Total Params to be less than 100k. 

## Folder Structure

* Experiments
  * Contains all the files that we experimented for finishing the assignment
* logs
  * Contains text files which has logs and summary for the model used
  * loss and accuracy graphs
* models
  * contains the model design
* utils
  * contains all utility methods needed for training and validating model
* main.py
  * Main file which calls the required methods sequentially just like colab notebook

## Data Augumentation:-
We have used below augmentation techniques using albumentation library
   1. horizontal flip of 0.3
   2. shiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101, always_apply=False, p=0.5)
   3. coarseDropout (max_holes = 1, max_height=16, max_width=16, min_holes = 1, min_height=16, min_width=16, fill_value=(0.4914, 0.4822, 0.4465), mask_fill_value = None)

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
## Model:-

## Training Logs:-

| Params Count | Best Train Accuracy | Best Test Accuracy | Link to Logs                                                 |
| ------------------- | ----------------- | ----------------- | ------------ | ------------------- | ------------------ | ------------------------------------------------------------ |
| 99,968        | 75.02               | 86.47           | [Training Logs](https://github.com/NSR9/Extensive-Vision-AI/blob/main/Assignment_7/logs) |

## Graphs and Plots :-

| Graph 1: Training Loss  | Graph 2: Test/Validation Loss      |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![image](https://user-images.githubusercontent.com/51078583/121725761-4bb08000-cb07-11eb-98de-296e91f6a74b.png) | ![image](https://user-images.githubusercontent.com/51078583/121725803-59fe9c00-cb07-11eb-818f-ca5cb510792d.png) |

## Contributors:-
<<<<<<< HEAD

=======
>>>>>>> c45bea554b97fc893f91b83d16ebb61ce8ac2066
1. Avinash Ravi
2. Nandam Sriranga Chaitanya
3. Saroj Raj Das
4. Ujjwal Gupta
