# Assignemnt 7
## Problem statement:-

## WorkFlow:-
### Enchancements used:-

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
## Model:-

## Training Logs:-

## Contributors:-
1. Avinash Ravi
2. Nandam Sriranga Chaitanya
3. Saroj Raj Das
4. Ujjwal Gupta
