import torch
import torchvision as tvison
from torch import nn, optim
import numpy as np
from models.PClassUtils import D_Block, FromRGB

class Encoder(nn.Module):
    """Encoder

    Args:
        nn (_type_): _descriptionAchei nu_
    """

    def __init__(self, latent_dim, channel_num, in_resolution_dim):
        super(Encoder, self).__init__()

        #Depth of network
        self.depth = 1
        #Degree of network interpolation of previous and current output
        self.alpha = 1
        self.fade_iters = 0
        #Downsampling the output of the previous networks
        self.downsample = nn.AvgPool2d(kernel_size=(2,2), stride=(2,2))
        #Reshaping the output of the previous networks
        self.reshape = nn.Flatten()
        self.drop = nn.Dropout(0.2)
        #List of networks with growing progressively the output
        self.current_net = nn.ModuleList(
                                []
                                )
        #Network with transforming the output in RGB image
        self.fromRGB = nn.ModuleList(
                                []
                                )
        #Network with transforming MU
        flatten_result = int(in_resolution_dim *(2**(np.log2(in_resolution_dim)-2)))*4*4
        self.dense_mu = nn.Linear(flatten_result, latent_dim)
        #Network with transforming MU
        self.log_sigma = nn.Linear(flatten_result, latent_dim)
       

        for d in reversed(range(2,int(np.log2(in_resolution_dim)))): # from 2 to number that pow 2 equals the out_resolution_dim
    
            in_channels = int((in_resolution_dim*(2**(d-1) ))/2)
            out_channels = int(in_resolution_dim*(2**(d-1) ))

            self.fromRGB.append(FromRGB(channel_num, int(out_channels/2)))
            self.current_net.append(D_Block(in_channels, out_channels))


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
 
        
    def forward(self, input):
        #print(input.shape)
        x = self.fromRGB[self.depth-1](input)
        #print(x.shape)
        x = self.current_net[self.depth-1](x)
        #print(x.shape)
        if self.alpha <1:

            x_rgb = self.downsample(input)
            #print(x_rgb.shape)
            x_old = self.fromRGB[self.depth-2](x_rgb)
            #print(x_old.shape)
            x = (1-self.alpha)* x_old + self.alpha *x
            #print(x.shape)
            self.alpha += self.fade_iters
        
        for block in reversed(self.current_net[:self.depth-1]):
            x =  block(x)
            #print(x.shape)

        x = self.reshape(x)
        #print(x.shape)

        mu = self.dense_mu(x)
        #print(mu.shape)
        log_sigma = self.log_sigma(x)
        #print(log_sigma.shape)

        return mu, log_sigma
      
    def growing_net(self, num_iters):
        self.fade_iters = 1/num_iters
        self.alpha = 1/num_iters

        self.depth +=1
