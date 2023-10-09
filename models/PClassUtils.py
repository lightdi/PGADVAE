import torch
import numpy as np
import torchvision as tvison
from torch import nn, optim

class ToRGB(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        to_rgb = [
            
                 #ToRGB
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(1, 1),stride=(1,1)),

                #Finlize
                nn.Tanh()
        ]
        self.to_rgb = nn.Sequential(*to_rgb)
        for m in self.modules():
            if  isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)

            elif isinstance(m, nn.ConvTranspose2d):
                pass
                m.weight.data.normal_(0, 0.02)
            
            elif isinstance(m, nn.Linear):
                pass
                m.weight.data.normal_(0, 0.02)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

    def forward(self,x):
        x = self.to_rgb(x)
        return x


class G_Block(nn.Module):
    
    def __init__(self, in_channels, out_channels):

        super().__init__()    
       
        gen = [
            
                #UpBlock
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(4,4), stride=2, bias=False, padding=1),
                nn.BatchNorm2d(out_channels, momentum=0.9),
                nn.ReLU(),
        ]

        self.gen = nn.Sequential(*gen)
        self.drop = nn.Dropout(0.2)
        
        for m in self.modules():
            if  isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)

            elif isinstance(m, nn.ConvTranspose2d):
                pass
                m.weight.data.normal_(0, 0.02)
            
            elif isinstance(m, nn.Linear):
                pass
                m.weight.data.normal_(0, 0.02)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

    def forward(self,x):
      
        x = self.gen(x)
        #x = self.drop(x)
     
        return x

class FromRGB(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        from_rgb = [
            
                 #ToRGB
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1),stride=(1,1)),
                nn.BatchNorm2d(out_channels,momentum=0.9),
                nn.LeakyReLU(),

        ]
        self.from_rgb = nn.Sequential(*from_rgb)

        for m in self.modules():
            if  isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)

            elif isinstance(m, nn.ConvTranspose2d):
                pass
                m.weight.data.normal_(0, 0.02)
            
            elif isinstance(m, nn.Linear):
                pass
                m.weight.data.normal_(0, 0.02)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

    
    def forward(self,x):
        x = self.from_rgb(x)
        return x


class D_Block(nn.Module):
    
    def __init__(self, in_channels, out_channels):

        super().__init__()    
        
        disc = [
                        #DownBlock
                        nn.Conv2d(in_channels, out_channels,kernel_size=(4,4), padding=1, stride=2, bias=False,),
                        nn.BatchNorm2d(out_channels, momentum=0.9),
                        nn.LeakyReLU(),
                    ]
   
        self.disc = nn.Sequential(*disc)
        self.drop = nn.Dropout(0.2)
        
        for m in self.modules():
            if  isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)

            elif isinstance(m, nn.ConvTranspose2d):
                pass
                m.weight.data.normal_(0, 0.02)
            
            elif isinstance(m, nn.Linear):
                pass
                m.weight.data.normal_(0, 0.02)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                nn.init.constant_(m.bias.data, 0)


    def forward(self,x):
      
        x = self.disc(x)
        #x = self.drop(x)
     
        return x

class PView(nn.Module):
    def __init__(self, channels, size):
        super().__init__()
        self.channels = channels
        self.size = size

    def forward(self,x):
        return x.view(-1, self.channels, self.size, self.size)