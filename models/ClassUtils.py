import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.parameter import Parameter


#Utils class

class EqualizedLR_Conv2d(nn.Module):
    def __init__(self,in_channel,out_channel, kernel_size, stride=1, padding=0):
        super().__init__()
        self.padding = padding
        self.stride = stride
        self.scale = np.sqrt(2/(in_channel * kernel_size[0] * kernel_size[1]))

        self.weight = Parameter(torch.Tensor(out_channel, in_channel, *kernel_size))
        self.bias = Parameter(torch.Tensor(out_channel))

        nn.init.normal_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self,x):
        return F.conv2d(x, self.weight * self.scale, self.bias, self.stride, self.padding)
        

class Pixel_norm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a):
        b =  a/torch.sqrt(torch.sum(a**2, dim=1, keepdim=True)+ 10e-8)
        return b


class Minibatch_std(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        size = list(x.size())
        size[1] = 1

        std = torch.std(x, dim=0)
        mean = torch.mean(std)

        return torch.cat((x,mean.repeat(size)),dim=1)



# suport class

class FromRGB(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = EqualizedLR_Conv2d(in_channel, out_channel, kernel_size=(1, 1),stride=(1,1))
        self.relu = nn.LeakyReLU(0.2)

    def forward (self, x):
        x = self.conv(x)
        return self.relu(x)


class ToRGB(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = EqualizedLR_Conv2d(in_channel, out_channel, kernel_size=(1,1), stride=(1,1))

    def forward(self, x):
        return self.conv(x)


class G_Block(nn.Module):
    def __init__(self, in_channel, out_channel, initial_block=False):
        super().__init__()
        if initial_block:
            self.upsample = None
            self.conv1 = EqualizedLR_Conv2d(in_channel,out_channel,kernel_size=(4,4),stride=(1,1), padding=(3,3))
        else:
            self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
            self.conv1 = EqualizedLR_Conv2d(in_channel,out_channel,kernel_size=(3,3),stride=(1,1), padding=(1,1))
        self.conv2 = EqualizedLR_Conv2d(in_channel, out_channel,kernel_size=(3,3),stride=(1,1),padding=(1,1))
        self.relu = nn.LeakyReLU(0.2)
        self.pixelwisenorm = Pixel_norm()
        nn.init.normal_(self.conv1.weight)
        nn.init.normal_(self.conv2.weight)
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)

    def forward(self,x):

        if self.upsample is not None:
                x = self.upsample(x)
        x = self.conv1(x)
        x = self.relu(x)
        x= self.pixelwisenorm(x)
        x = self.conv2(x)
        x = self.relu(x)
        x= self.pixelwisenorm(x)
        return x


class D_Block(nn.Module):
    def __init__(self,in_channel, out_channel,initial_block=False):
        super().__init__()
        
        if initial_block:
            self.minibatchstd = Minibatch_std()
            self.conv1 = EqualizedLR_Conv2d(in_channel+1, out_channel,kernel_size=(3,3),stride=(1,1), padding=(1,1))
            self.conv2 = EqualizedLR_Conv2d(in_channel, out_channel,kernel_size=(4,4),stride=(1,1))
            self.outlayer = nn.Sequential (
                                            nn.Flatten(),
                                            nn.Linear(out_channel, 1)
                                        )

        else:
            self.minibatchstd = None
            self.conv1 = EqualizedLR_Conv2d(in_channel, out_channel, kernel_size=(3,3), stride=(1,1), padding=(1,1))
            self.conv2 = EqualizedLR_Conv2d(in_channel, out_channel, kernel_size=(3,3), stride=(1,1), padding=(1,1))
            self.outlayer = nn.AvgPool2d(kernel_size=(2,2),stride=(2,2))

        self.relu = nn.LeakyReLU(0.2)
        nn.init.normal_(self.conv1.weight)
        nn.init.normal_(self.conv2.weight)
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)

    def forward(self,x):

        if self.minibatchstd is not None:
            x = self.minibatchstd(x)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.outlayer(x)
        return x
        
