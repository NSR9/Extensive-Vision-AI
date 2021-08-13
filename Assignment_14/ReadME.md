# Assignment 14 - DETR End-to-End Object Detection with Transformers

1. Take a look at this post (Links to an external site.), which explains how to fine-tune DETR on a custom dataset. 
2. Replicate the process and train the model yourself. Everything (Links to an external site.) is mentioned in the post. The objectives are:
    - to understand how fine-tuning works
    - to understand architectural related concepts


# What is DETR:

The main ingredients of DEtection TRansformer or DETR, are a set-based global loss that forces unique predictions via bipartite matching, and a transformer encoder-decoder architecture. Given a fixed small set of learned object queries, DETR reasons about the relations of the objects and the global image context to directly output the final set of predictions in parallel.

## What’s Good
Compares to previous state-of-the-art models in object detection, DETR has significantly less hyperparameter to set. DETR doesn’t need to set the number of anchor box, aspect ratio, default cordinates of bounding boxes, even threshold for non-maximum surpression. DETR hand all those task to encoder-decoder transformer and bipartite matching, and achieve more general models for diversified usage.
























