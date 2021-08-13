# Assignment 13 - ViT with Transformers

1. Let's review this blog  (Links to an external site.)on using ViT for Cats vs Dogs. Your assignment is to implement this blog and train the ViT model for Cats vs Dogs. If you wish you can use transfer learning.
2. Share the link to the README that describes your CATS vs DOGS training using VIT. Expecting to see the training logs (at least 1) there.  
Share the link to the notebook where I can find your Cats vs Dogs Training
Expecting a Separate or same README to explain your understanding of all the Classes that we covered in the class. 


## ViT Classes As Explained:

1. class PatchEmbeddings

class PatchEmbeddings(nn.Module):
    """
    Image to Patch Embedding.

    """

    def __init__(self, image_size=224, patch_size=16, num_channels=3, embed_dim=768):
        super().__init__()
        image_size = to_2tuple(image_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.projection = nn.Conv2d(num_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values):
        batch_size, num_channels, height, width = pixel_values.shape
        # FIXME look at relaxing size constraints
        if height != self.image_size[0] or width != self.image_size[1]:
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
            )
        x = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return x

This class takes the input image and converts into patch embeddings and then convert them to 1D arrary for further processing.



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

























