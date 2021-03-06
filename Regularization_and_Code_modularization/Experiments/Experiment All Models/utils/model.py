import torch
import torch.nn as nn
import torch.nn.functional as F

dropout_value = 0.03


class Group(nn.Module):
    def __init__(self):
        super(Group, self).__init__()
        # Input Block
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False
            ),  # Input 28x28 output 26x26 RF : 3x3
            nn.ReLU(),
            nn.GroupNorm(num_groups=2, num_channels=8),
            nn.Dropout(dropout_value),
            nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),  # Input 26x26 output 24x24 RF : 5x5
            nn.ReLU(),
            nn.GroupNorm(num_groups=4, num_channels=16),
            nn.Dropout(dropout_value),
        )

        # Transition Block
        self.trans1 = nn.Sequential(
            nn.MaxPool2d(2, 2),  #  Input 24x24 output 12x12 RF : 6x6
            nn.Conv2d(
                in_channels=16,
                out_channels=8,
                kernel_size=(1, 1),
                padding=0,
                bias=False,
            ),  # Input 12x12 output 12x12 RF : 6x6
        )

        # CONVOLUTION BLOCK 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),  # Input 12x12 output 10x10 RF : 6x6
            nn.ReLU(),
            nn.GroupNorm(num_groups=4, num_channels=16),
            nn.Dropout(dropout_value),
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),  # Input 10x10 output 8x8 RF : 10x10
            nn.ReLU(),
            nn.GroupNorm(num_groups=4, num_channels=16),
            nn.Dropout(dropout_value),
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),  # Input 8x8 output 6x6 RF : 14x14
            nn.ReLU(),
            nn.GroupNorm(num_groups=4, num_channels=16),
            nn.Dropout(dropout_value),
        )

        # OUTPUT BLOCK
        self.avgpool2d = nn.AvgPool2d(kernel_size=6)

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=(1, 1),
                padding=0,
                bias=False,
            ),  # Input 6x6 output 6x6 RF : 18x18
            nn.ReLU(),
            nn.GroupNorm(num_groups=4, num_channels=16),
            nn.Dropout(dropout_value),
        )

        self.conv4 = nn.Conv2d(
            in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False
        )  # Input 6x6 output 6x6 RF : 18x18

    def forward(self, x):
        x = self.conv1(x)
        x = self.trans1(x)
        x = self.conv2(x)
        x = self.avgpool2d(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)


class Layer(nn.Module):
    def __init__(self):
        super(Layer, self).__init__()
        # Input Block
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False
            ),  # Input 28x28 output 26x26 RF : 3x3
            nn.ReLU(),
            nn.LayerNorm((8, 26, 26)),
            nn.Dropout(dropout_value),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),  # Input 26x26 output 24x24 RF : 5x5
            nn.ReLU(),
            nn.LayerNorm((16, 24, 24)),
            nn.Dropout(dropout_value),
        )

        # Transition Block
        self.trans1 = nn.Sequential(
            nn.MaxPool2d(2, 2),  #  Input 24x24 output 12x12 RF : 6x6
            nn.Conv2d(
                in_channels=16,
                out_channels=8,
                kernel_size=(1, 1),
                padding=0,
                bias=False,
            ),  # Input 12x12 output 12x12 RF : 6x6
        )

        # CONVOLUTION BLOCK 2
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),  # Input 12x12 output 10x10 RF : 6x6
            nn.ReLU(),
            nn.LayerNorm((16, 10, 10)),
            nn.Dropout(dropout_value),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),  # Input 10x10 output 8x8 RF : 10x10
            nn.ReLU(),
            nn.LayerNorm((16, 8, 8)),
            nn.Dropout(dropout_value),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),  # Input 8x8 output 6x6 RF : 14x14
            nn.ReLU(),
            nn.LayerNorm((16, 6, 6)),
            nn.Dropout(dropout_value),
        )

        # OUTPUT BLOCK
        self.avgpool2d = nn.AvgPool2d(kernel_size=6)

        self.conv6 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=(1, 1),
                padding=0,
                bias=False,
            ),  # Input 6x6 output 6x6 RF : 18x18
            nn.ReLU(),
            nn.LayerNorm((16, 1, 1)),
            nn.Dropout(dropout_value),
        )

        self.conv7 = nn.Conv2d(
            in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False
        )  # Input 6x6 output 6x6 RF : 18x18

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.trans1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avgpool2d(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)


class Batch(nn.Module):
    def __init__(self):
        super(Batch, self).__init__()
        # Input Block
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False
            ),  # Input 28x28 output 26x26 RF : 3x3
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(dropout_value),
            nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),  # Input 26x26 output 24x24 RF : 5x5
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value),
        )

        # Transition Block
        self.trans1 = nn.Sequential(
            nn.MaxPool2d(2, 2),  #  Input 24x24 output 12x12 RF : 6x6
            nn.Conv2d(
                in_channels=16,
                out_channels=8,
                kernel_size=(1, 1),
                padding=0,
                bias=False,
            ),  # Input 12x12 output 12x12 RF : 6x6
        )

        # CONVOLUTION BLOCK 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),  # Input 12x12 output 10x10 RF : 6x6
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value),
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),  # Input 10x10 output 8x8 RF : 10x10
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value),
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),  # Input 8x8 output 6x6 RF : 14x14
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value),
        )

        # OUTPUT BLOCK
        self.avgpool2d = nn.AvgPool2d(kernel_size=6)

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=(1, 1),
                padding=0,
                bias=False,
            ),  # Input 6x6 output 6x6 RF : 18x18
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value),
        )

        self.conv4 = nn.Conv2d(
            in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False
        )  # Input 6x6 output 6x6 RF : 18x18

    def forward(self, x):
        x = self.conv1(x)
        x = self.trans1(x)
        x = self.conv2(x)
        x = self.avgpool2d(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
