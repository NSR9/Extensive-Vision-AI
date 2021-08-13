# Assignment 14 - DETR End-to-End Object Detection with Transformers

1. Take a look at this post (Links to an external site.), which explains how to fine-tune DETR on a custom dataset. 
2. Replicate the process and train the model yourself. Everything (Links to an external site.) is mentioned in the post. The objectives are:
    - to understand how fine-tuning works
    - to understand architectural related concepts


# What is DETR:

In DETR, object detection problem is modeled as a direct set prediction problem.The two novel components of the new framework, called DEtection TRansformer or DETR, are:-
- a set-based global loss that forces unique predictions via bipartite matching.
- a transformer encoder-decoder architecture.

The main ingredients of DEtection TRansformer or DETR, are a set-based global loss that forces unique predictions via bipartite matching, and a transformer encoder-decoder architecture. Given a fixed small set of learned object queries, DETR reasons about the relations of the objects and the global image context to directly output the final set of predictions in parallel.

![image](https://user-images.githubusercontent.com/51078583/129374219-94d11b88-6677-474a-bcdb-0954d11a7785.png)

**How DETR differs from other object detection methods?** 

DETR formulates the object detection task as an image-to-set problem. Given an image, the model must predict an unordered set (or list) of all the objects present, each represented by its class, along with a tight bounding box surrounding each one. Transformer acts as a reasoning agent between the image features and the prediction.



## What’s Good
Compares to previous state-of-the-art models in object detection, DETR has significantly less hyperparameter to set. DETR doesn’t need to set the number of anchor box, aspect ratio, default cordinates of bounding boxes, even threshold for non-maximum surpression. DETR hand all those task to encoder-decoder transformer and bipartite matching, and achieve more general models for diversified usage.

## Architecture

The overall DETR architecture is easy to understand. It contains three main components:

1. CNN Backbone
1. Encoder-Decoder transformer
1. Simple feed-forward network

![image](https://user-images.githubusercontent.com/51078583/129373984-9fceefe3-c15a-4fc6-b944-d0205c5459e5.png)

### CNN Backbone:

DETR uses a conventional CNN backbone to learn a 2D representation of an input image. The model flattens it and supplements it with a positional encoding before passing it into a transformer encoder. A transformer decoder then takes as input a small fixed number(N) of learned positional embeddings, which we call object queries, and additionally attends to the encoder output. We pass each output embedding of the decoder to a shared feed forward network (FFN) that predicts either a detection (class and bounding box) or a ∅(no object) class.

### Encoder-Decoder transformer:

It is very similar to the original transformer block with minute differences adjusted to this task.

![image](https://user-images.githubusercontent.com/51078583/129375281-2b6ac781-cbbc-4967-b5ac-adc0a5819375.png)

**1. ENCODER LAYER**

First, a 1x1 convolution reduces the channel dimension of the high-level activation map from C to a smaller dimension d, creating a new feature map d×H×W. The encoder expects a sequence as input so it is collapsed to one dimension, resulting in a d×HW feature map.

Each encoder layer has a standard architecture and consists of a multi-head self-attention module and a feed forward network (FFN). Since the transformer architecture is permutation-invariant, they supplement it with fixed positional encodings that are added to the input of each attention layer.

**2. DECODER LAYER**

The decoder follows the standard architecture of the transformer, transforming N embeddings of size d using multi-headed self- and encoder-decoder attention mechanisms. **The difference with the original transformer is that DETR model decodes the N objects in parallel at each decoder layer.**

#### Object queries

The N object queries are transformed into an output embedding by the decoder. They are then independently decoded into box coordinates and class labels by a feed forward network (FFN), resulting N final predictions. The decoder receives queries (initially set to zero), output positional encoding (object queries), and encoder memory, and produces the final set of predicted class labels and bounding boxes through multiple multi-head self-attention and decoder-encoder attention. The first self-attention layer in the first decoder layer can be skipped.

### Simple feed-forward network

The final prediction is computed by a 3-layer perceptron with ReLU activation function and hidden dimension d, and a linear projection layer. The FFN predicts the normalized centercoordinates, height and width of the box w.r.t. the input image, and the linear layer predicts the class label using a softmax function. Since we predict afixed-size set of N bounding boxes, where N is usually much larger than theactual number of objects of interest in an image, an additional special class label ∅ is used to represent that no object is detected within a slot. This classplays a similar role to the “background” class in the standard object detection
approaches

## Bipartite Matching Loss

Unlike other object detection models label bounding boxes (or point, like methods in object as points) by matching multiple bounding boxes to one ground truth box, DETR is using bipartite matching, which is one-vs-one matching.

![image](https://user-images.githubusercontent.com/51078583/129376512-2c745300-f227-4588-a936-1ccf9c4db1fd.png)

By performing one-vs-one matching, its able to significantly reduce low-quality predictions, and achieve eliminations of output reductions like NMS. Bipartite matching loss is designed based on Hungarian algorithm.

Each element i of the ground truth set can be seen as a yi= (ci,bi) where ci is the target class label (which may be∅) and bi∈[0,1] is a vector that has four attributes — normalized ground truth box center coordinates, height and width relative to the image size. For the prediction with index σ(i) we define probability of class ci as ˆpσ(i)(ci) and the predicted box as ˆbσ(i). The first part of loss takes care of class prediction and the second part is the loss for the box prediction. After receiving all matched pairs for the set, the next step is to compute the loss function, the Hungarian loss.


# FineTune DETR:

Looking into the assignment and fine tuning DETR for creatign bounding boxes around balloons , using pretrained weights . 























