# Assignment 12 - The Dawn of the Transformers:

## Problem Statement:

1. Implement that [Spatial Transformer Code](https://brsoff.github.io/tutorials/intermediate/spatial_transformer_tutorial.html) for CIFAR10 and submit. Must have proper readme and must be trained for 50 Epochs.
2. describe using text and your drawn images, the classes in this [FILE](https://github.com/jeonsworld/ViT-pytorch/blob/main/models/modeling.py) (Links to an external site.):
 - Block
 - Embeddings
 - MLP
 - Attention
 - Encoder
 
## Spatial Transformers(STN's):

Spatial Transformer Network (STN), by Google DeepMind helps to crop out and scale-normalizes the appropriate region, which can simplify the subsequent classification task and lead to better classification performance

A Spatial Transformer Network (STN) is a learnable module that can be placed in a Convolutional Neural Network (CNN), to increase the spatial invariance in an efficient manner. Spatial invariance refers to the invariance of the model towards spatial transformations of images such as rotation, translation and scaling. Invariance is the ability of the model to recognize and identify features even when the input is transformed or slightly modified. Spatial Transformers can be placed into CNNs to benefit various tasks. One example is image classification.

**The working of Spatial Transformer Network on the Distorted MNIST dataset can be seen as follows:**
![image](https://user-images.githubusercontent.com/51078583/127387882-ba6dda8c-304c-47fd-a64f-723f074395cd.png)


A STN is majorly divided into 3 parts :
- Localisation Net
- Grid Generator
- Sampler

Which can be visiualized using the following image:

![image](https://user-images.githubusercontent.com/51078583/127382590-c1f9ed10-2964-4829-a1c7-67580c3cec2e.png)

### 1. Localization Network

It is a simple neural network with a few convolution layers and a few dense layers. It predicts the parameters of transformation as output. These parameters determine the angle by which the input has to be rotated, the amount of translation to be done, and the scaling factor required to focus on the region of interest in the input feature map.
 
### 2. Grid Generator

The transformation parameters predicted by the localization net are used in the form of an affine transformation matrix of size 2 x 3 for each image in the batch. An affine transformation is one which preserves points, straight lines and planes. Parallel lines remain parallel after affine transformation. Rotation, scaling and translation are all affine transformations.

![image](https://user-images.githubusercontent.com/51078583/127388037-68615834-1b48-44a7-92fb-4abb266df9d8.png)

Here, (xti,yti) are the target coordinates of the target grid in the output feature map, (xsi,ysi) are the input coordinates in the input feature map, and Aθ is the affine transformation matrix. T is the transformation and A is the matrix representing the affine transformation. θ11, θ12, θ21, θ22 determine the angle by which the image has to be rotated. θ13, θ23 determine the translations along width and height of the image respectively. Thus we obtain a sampling grid of transformed indices.


### 3. Sampler

This is the last part of the spatial transformer network. We have the input feature map and also the parameterized sampling grid with us now. To perform the sampling, we give the feature map U and sampling grid Tθ(G) as input to the sampler. The sampling kernel is applied to the source coordinates using the parameters θ and we get the output V.

![image](https://user-images.githubusercontent.com/51078583/127388329-bcc89a75-6558-439d-bd25-52ad90d72415.png)

## CIFAR10 with STN:

Implement that [Spatial Transformer Code](https://brsoff.github.io/tutorials/intermediate/spatial_transformer_tutorial.html) for CIFAR10 and submit. Must have proper readme and must be trained for 50 Epochs.
- [COLAB LINK TO THE CODE]()
- [GITHUB LINK TO THE CODE]()

### Model Architecture:

### Training Logs For Cifar10:

### Results:

## Vision Transformers(ViT's):

## Code Block Explanation:

### Block:

The block is bacically that part of the code which combines both the results of attention and MLP forming a skip kind of connection. 

According to the code:
- The input send is passed on to Layer Normalization of *(onfig.hidden_size, eps=1e-6)* which is in turn passed to the attention layer. The output from the attention layer and the impage before the Layer Normaliztion is clubbed together or added and passed in for the input or next step. 
- As a part of the next step the output coming in after the attention and the skip connection is again passed to the Layer Normalization of *(config.hidden_size, eps=1e-6)*. The output of Layer Normalization is then passed to the Feed Forword network which in this case is the MLP. The output of MLP and the input data before the Layer Normalization of MLP is again clubbed and as a result send as the final output. 

![image](https://user-images.githubusercontent.com/51078583/127564671-a62ca0e0-4b3d-4741-9d3a-e009fc0f9d8d.png)

```
    class Block(nn.Module):
        def __init__(self, config, vis):
            super(Block, self).__init__()
            self.hidden_size = config.hidden_size
            self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
            self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
            self.ffn = Mlp(config)
            self.attn = Attention(config, vis)

        def forward(self, x):
            h = x
            x = self.attention_norm(x)
            x, weights = self.attn(x)
            x = x + h

            h = x
            x = self.ffn_norm(x)
            x = self.ffn(x)
            x = x + h
            return x, weights
```
### Embeddings:

- Embedding being the first part of the  Vision Transformers structure includes basic division of the images into patches before it can be used for encoding.

![image](https://user-images.githubusercontent.com/51078583/127558337-030dc621-4e8c-4211-a61c-cad064a91a19.png)

In the below code something similar is happening . 

- To study the image ViT divides an image into a grid of square patches. Each patch is flattened into a single vector by concatenating the channels of all pixels in a patch and then linearly projecting it to the desired input dimension. 
- These patches are represented in a linear layer. Conv2D layer in the _init_ funtion performs the reprentation of the patches with the followinf details *(in_channels=in_channels, out_channels=config.hidden_size, kernel_size=patch_size,=patch_size)* . The Above code is majorly concerned with the creation of patches based on the grids. 
- To have the model know which patch comes after what there is a positional embedding for these patches, *nn.Parameter(torch.zeros(1, n_patches+1, config.hidden_size))*. To make this job easier and more perfect we let the neural network decide numbering/positioning refernce for the patches for the image taken in so that the neural netowkr comes up with the most optimum and correct representation . 
- The forward part of the function is basically for moving thru the initiailzation of the class for the input values. 

```
class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        img_size = _pair(img_size)

        if config.patches.get("grid") is not None:
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            n_patches = (img_size[0] // 16) * (img_size[1] // 16)
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers,
                                         width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, config.hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])
        
    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        if self.hybrid:
            x = self.hybrid_model(x)
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = torch.cat((cls_tokens, x), dim=1)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

```

### MLP:

MLP-Mixer consists of per-patch linear embeddings, Mixer layers, and a classifier head. Mixer layers contain one token-mixing MLP and one channel-mixing MLP, each consisting of two fully-connected layers and a GELU(Gaussian Error Linear Unit) nonlinearity. 

The Output from the Attention block is passed onto the MLP block which can be represented by the following diagram :

![image](https://user-images.githubusercontent.com/51078583/127562674-a71fd4d0-f5aa-4d41-9764-211cc538122f.png)


All image patches are projected linearly with the same projection matrix provides a better interaction amongst the image to learn . 

#### GeLU(Gaussian Error Linear Unit) :


GeLU performs element-wise activation function on a given input tensor. An activation function used in the most recent Transformers – Google's BERT and OpenAI's GPT-2. The paper is from 2016, but is only catching attention up until recently.

![image](https://user-images.githubusercontent.com/51078583/127562071-fb6de6c1-5b16-4c7f-a98a-12a5557887de.png)

So it's just a combination of some functions (e.g. hyperbolic tangent tanh) and approximated numbers – there is not much to say about it. What is more interesting, is looking at the graph for the gaussian error linear unit:

![image](https://user-images.githubusercontent.com/51078583/127562186-ab0a67e0-899b-44b3-913b-deb8ea0a74de.png)

```
 class Mlp(nn.Module):
        def __init__(self, config):
            super(Mlp, self).__init__()
            self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
            self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
            self.act_fn = ACT2FN["gelu"]
            self.dropout = Dropout(config.transformer["dropout_rate"])

            self._init_weights()

        def _init_weights(self):
            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.xavier_uniform_(self.fc2.weight)
            nn.init.normal_(self.fc1.bias, std=1e-6)
            nn.init.normal_(self.fc2.bias, std=1e-6)

        def forward(self, x):
            x = self.fc1(x)
            x = self.act_fn(x)
            x = self.dropout(x)
            x = self.fc2(x)
            x = self.dropout(x)
            return x
 ```

### Attention

### Encoder

## Refernce Link:

- [Spatial Transformers](https://towardsdatascience.com/review-stn-spatial-transformer-network-image-classification-d3cbd98a70aa)
- [Vision Transformers](https://youtu.be/4Bdc55j80l8)
- [MLP](https://medium.com/@nabil.madali/an-all-mlp-architecture-for-vision-7e7e1270fd33)

## Contributors:
