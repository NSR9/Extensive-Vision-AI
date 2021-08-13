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

![image](https://user-images.githubusercontent.com/51078583/126815847-a91f9a2d-4bd8-4fdc-8ad2-59b55d3d0a8c.png)


## Training logs:

























