# Assignment 13 - ViT with Transformers

1. Let's review this blog  (Links to an external site.)on using ViT for Cats vs Dogs. Your assignment is to implement this blog and train the ViT model for Cats vs Dogs. If you wish you can use transfer learning.
2. Share the link to the README that describes your CATS vs DOGS training using VIT. Expecting to see the training logs (at least 1) there.  
Share the link to the notebook where I can find your Cats vs Dogs Training
Expecting a Separate or same README to explain your understanding of all the Classes that we covered in the class. 


# What is DETR:

The main ingredients of DEtection TRansformer or DETR, are a set-based global loss that forces unique predictions via bipartite matching, and a transformer encoder-decoder architecture. Given a fixed small set of learned object queries, DETR reasons about the relations of the objects and the global image context to directly output the final set of predictions in parallel.

## What’s Good
Compares to previous state-of-the-art models in object detection, DETR has significantly less hyperparameter to set. DETR doesn’t need to set the number of anchor box, aspect ratio, default cordinates of bounding boxes, even threshold for non-maximum surpression. DETR hand all those task to encoder-decoder transformer and bipartite matching, and achieve more general models for diversified usage.

## Data Augumentation:

DataSet: Tiny Imagenet has 200 classes of 64x64 images, each class having 500 images each.

- Random crop after applying padding of (min_height=40, min_width=40, always_apply=True)
- Horizontal Flip 
- Coarse Dropout
- Normalize

![image](https://user-images.githubusercontent.com/51078583/126819514-3163317d-cfbd-4e4b-bad7-7f2638347dbf.png)

**Used One cylce LR**

## Model Summary:


## Training logs:

100%
313/313 [1:29:20<00:00, 17.13s/it]

Epoch : 1 - loss : 0.6955 - acc: 0.5068 - val_loss : 0.6966 - val_acc: 0.4970

100%
313/313 [1:14:26<00:00, 14.27s/it]

Epoch : 2 - loss : 0.6907 - acc: 0.5255 - val_loss : 0.6854 - val_acc: 0.5562

100%
313/313 [15:00<00:00, 2.88s/it]

Epoch : 3 - loss : 0.6853 - acc: 0.5477 - val_loss : 0.6806 - val_acc: 0.5508

100%
313/313 [44:53<00:00, 8.60s/it]

Epoch : 4 - loss : 0.6758 - acc: 0.5795 - val_loss : 0.6732 - val_acc: 0.5862

100%
313/313 [29:46<00:00, 5.71s/it]

Epoch : 5 - loss : 0.6752 - acc: 0.5734 - val_loss : 0.6659 - val_acc: 0.5938

100%
313/313 [14:49<00:00, 2.84s/it]

Epoch : 6 - loss : 0.6664 - acc: 0.5887 - val_loss : 0.6750 - val_acc: 0.5698

100%
313/313 [13:14<00:00, 2.54s/it]

Epoch : 7 - loss : 0.6563 - acc: 0.6005 - val_loss : 0.6502 - val_acc: 0.6036

100%
313/313 [29:53<00:00, 5.73s/it]

Epoch : 8 - loss : 0.6457 - acc: 0.6183 - val_loss : 0.6508 - val_acc: 0.6078

100%
313/313 [14:55<00:00, 2.86s/it]

Epoch : 9 - loss : 0.6391 - acc: 0.6270 - val_loss : 0.6448 - val_acc: 0.6175

100%
313/313 [2:41:19<00:00, 30.93s/it]

Epoch : 10 - loss : 0.6319 - acc: 0.6324 - val_loss : 0.6295 - val_acc: 0.6292

100%
313/313 [29:23<00:00, 5.63s/it]

Epoch : 11 - loss : 0.6255 - acc: 0.6417 - val_loss : 0.6308 - val_acc: 0.6406

100%
313/313 [14:31<00:00, 2.79s/it]

Epoch : 12 - loss : 0.6199 - acc: 0.6499 - val_loss : 0.6249 - val_acc: 0.6466

























