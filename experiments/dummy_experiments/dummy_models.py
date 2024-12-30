import torch
import torch.nn as nn

from global_utils.model_operations import transform_to_sequential, split_model_in_two


# Define a block for processing image input (convolutional layer with activation and flattening)
class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=1, padding=1)
        self.activation = nn.ReLU()
        self.flatten = nn.Flatten()  # Flatten the output to prepare for fully connected layers

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        x = self.flatten(x)
        return x


# Define a block for fully connected processing (linear layer with activation)
class FullyConnectedBlock(nn.Module):
    def __init__(self, input_size, output_size):
        super(FullyConnectedBlock, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        return x


# Combine the blocks into the final model
class TwoBlockModel(nn.Module):
    def __init__(self, input_channels, conv_output_channels, output_size):
        super(TwoBlockModel, self).__init__()
        self.model = nn.Sequential(
            ConvBlock(input_channels, conv_output_channels),  # First block: Convolutional block with flattening
            FullyConnectedBlock(conv_output_channels * 224 * 224, output_size)  # Second block: Fully connected block
        )

    def forward(self, x):
        return self.model(x)

def get_sequential_two_block_model():
    input_channels = 3
    conv_output_channels = 16  # Number of output channels for the convolutional layer
    output_size = 2  # Number of classes for output

    model = TwoBlockModel(input_channels, conv_output_channels, output_size)
    seq_model = transform_to_sequential(model)

    return seq_model

if __name__ == '__main__':
    # Example of using the model
    input_channels = 3
    conv_output_channels = 16  # Number of output channels for the convolutional layer
    output_size = 2  # Number of classes for output

    model = TwoBlockModel(input_channels, conv_output_channels, output_size)
    print(model)
    seq_model = transform_to_sequential(model)
    first, second = split_model_in_two(seq_model, 1)
    print(first)
    print(second)

    # Dummy input
    x = torch.randn(8, 3, 224, 224)  # Batch size of 8, image size 3x224x224
    output = model(x)
    print(output.shape)  # Should output shape (8, 10)
