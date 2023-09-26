'''
Build and test a simple three-layer neural net in PyTorch

Adapted from

  https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
'''

import torch
from torch import nn


device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)

image = torch.rand(10, 1, 28, 28)

output = model(image)

print(output)
