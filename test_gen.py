import torch
import torchvision as tvison
from torch import nn, optim
from generator import Generator


g = Generator(100, 3, 64)

x = torch.tensor([[1.0, 1.0, 1.0,1.0, 1.0, 1.0,1.0, 1.0, 1.0,1.0,
                  1.0, 1.0, 1.0,1.0, 1.0, 1.0,1.0, 1.0, 1.0,1.0,
                  1.0, 1.0, 1.0,1.0, 1.0, 1.0,1.0, 1.0, 1.0,1.0,
                  1.0, 1.0, 1.0,1.0, 1.0, 1.0,1.0, 1.0, 1.0,1.0,
                  1.0, 1.0, 1.0,1.0, 1.0, 1.0,1.0, 1.0, 1.0,1.0,
                  1.0, 1.0, 1.0,1.0, 1.0, 1.0,1.0, 1.0, 1.0,1.0,
                  1.0, 1.0, 1.0,1.0, 1.0, 1.0,1.0, 1.0, 1.0,1.0,
                  1.0, 1.0, 1.0,1.0, 1.0, 1.0,1.0, 1.0, 1.0,1.0,
                  1.0, 1.0, 1.0,1.0, 1.0, 1.0,1.0, 1.0, 1.0,1.0,
                  1.0, 1.0, 1.0,1.0, 1.0, 1.0,1.0, 1.0, 1.0,1.0    
    ]])

print (x)

x_result = g(x)

print(x_result)
print(x_result.shape)

