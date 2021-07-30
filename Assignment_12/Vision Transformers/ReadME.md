## Vision Transformers(ViT's):
The field of Computer Vision has for years been dominated by Convolutional Neural Networks (CNNs).
But recently this field has been incredibly revolutionized by the architecture of Vision Transformers (ViT), which through the mechanism of self-attention has proven to obtain excellent results on many tasks.




## Code Block Explanation:

### Block:

The block is bacically that part of the code which combines both the results of attention and MLP forming a skip kind of connection. 

According to the code:
- The input send is passed on to Layer Normalization of *(onfig.hidden_size, eps=1e-6)* which is in turn passed to the attention layer. The output from the attention layer and the impage before the Layer Normaliztion is clubbed together or added and passed in for the input or next step. 
- As a part of the next step the output coming in after the attention and the skip connection is again passed to the Layer Normalization of *(config.hidden_size, eps=1e-6)*. The output of Layer Normalization is then passed to the Feed Forword network which in this case is the MLP. The output of MLP and the input data before the Layer Normalization of MLP is again clubbed and as a result send as the final output. 

![image](https://user-images.githubusercontent.com/51078583/127666429-501e735a-7169-4b97-9659-0b3363c178d5.png)

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

Embedding being the first part of the  Vision Transformers structure includes basic division of the images into patches before it can be used for encoding. As discussed earlier, an image is divided into small patches here let’s say 9, and each patch might contain 16×16 pixels.  The input sequence consists of a flattened vector ( 2D to 1D ) of pixel values from a patch of size 16×16. Each flattened element is fed into a linear projection layer that will produce what they call the “Patch embedding”. Position embeddings are added to the patch embeddings to retain positional information.

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

![image](https://user-images.githubusercontent.com/51078583/127666511-c771e3a0-fef7-4501-b401-7c2ef0c8914b.png)


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

The Attention mechanism enables the transformers to have extremely long term memory. A transformer model can “attend” or “focus” on all previous tokens that have been generated.The attention takes three inputs, the famous queries, keys, and values, and computes the attention matrix using queries and values and use it to “attend” to the values.

- To achieve self-attention, we feed the input into 3 distinct fully connected layers to create the query, key, and value vectors.
- After feeding the query, key, and value vector through a linear layer, the queries and keys undergo a dot product matrix multiplication to produce a score matrix.

![image](https://user-images.githubusercontent.com/51078583/127671621-38a3dbe3-d4b5-45d8-bc84-b49d8ec68b7c.png)

- The score matrix determines how much focus should patch have. So each patch will have a score that corresponds to other patches in the time-step. The higher the score the more focus. This is how the queries are mapped to the keys.
- Then, the scores get scaled down by getting divided by the square root of the dimension of query and key. This is to allow for more stable gradients, as multiplying values can have exploding effects.
<img src="https://user-images.githubusercontent.com/51078583/127671069-2a9e54d4-6adf-4713-8023-326a732fe8eb.png" alt="Girl in a jacket" width="250" height="250">
- Next, you take the softmax of the scaled score to get the attention weights, which gives you probability values between 0 and 1. By doing a softmax the higher scores get heighten, and lower scores are depressed. This allows the model to be more confident about which patches to attend too.
- Then you take the attention weights and multiply it by your value vector to get an output vector. The higher softmax scores will keep the value of those patches the model learns is more important. The lower scores will drown out the irrelevant patches.

![image](https://user-images.githubusercontent.com/51078583/127670857-1c20c5d8-ca4a-4b6d-b0a5-2b6fb058e467.png)

Here is the overall representation of the attention block of the code(block diagram):

![image](https://user-images.githubusercontent.com/51078583/127671389-2eff966e-f3e9-4daf-8bbb-e9fb8f6b6803.png)

```
class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights

```
### Encoder

Transformers consists of Encoder - Decoder Block. But apparently in ViT's we have only the Encoder block. 

The Sole purpose of this as seen in the code is to take multiple attention in a Multi-head attention framework and Multi-Layer Perceptron (MLP) framework and combine them in a sequential Manner. 

![image](https://user-images.githubusercontent.com/51078583/127673545-49a4cd8f-c381-4b08-87c3-f52bc201a820.png)

In the code below the input is and is passed to the multi-layer and attention for the same is calculated for each layer. It is then combined and normalized and send as an output. 

```
    class Encoder(nn.Module):
        def __init__(self, config, vis):
            super(Encoder, self).__init__()
            self.vis = vis
            self.layer = nn.ModuleList()
            self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
            for _ in range(config.transformer["num_layers"]):
                layer = Block(config, vis)
                self.layer.append(copy.deepcopy(layer))

        def forward(self, hidden_states):
            attn_weights = []
            for layer_block in self.layer:
                hidden_states, weights = layer_block(hidden_states)
                if self.vis:
                    attn_weights.append(weights)
            encoded = self.encoder_norm(hidden_states)
            return encoded, attn_weights

```

## Refernce Link:

- [Vision Transformers](https://youtu.be/4Bdc55j80l8)
- [MLP](https://medium.com/@nabil.madali/an-all-mlp-architecture-for-vision-7e7e1270fd33)
- [Transformers](https://towardsdatascience.com/illustrated-guide-to-transformers-step-by-step-explanation-f74876522bc0)

## Contributors:
1. Avinash Ravi
2. Nandam Sriranga Chaitanya
3. Saroj Raj Das
4. Ujjwal Gupta

