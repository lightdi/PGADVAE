import torch
import numpy as np
from models.PClassUtils import G_Block, ToRGB, PView
import torchvision as tvison
from torch import nn, optim


class Decoder(nn.Module):
    """
         Decoder    
    """
    def __init__(self, latent_dim, channel_num, out_resolution_dim):
        """ Class to generate images from progressive way

        Args:
            latent_dim (int): size of latent dimension
            channel_num (int): number of channels to generate
            out_resolution_dim (int): Size of out image generated
            noise_dim (int, optional): Size of noise to add in generated image. Defaults to 50.
        """        
        super().__init__()
        self.out_resolution_dim = out_resolution_dim
        #Transform 128x128 to 96x96 image

         #Depth of network
        self.depth = 1
        #Degree of network interpolation of previous and current output
        self.alpha = 1
        self.fade_iters = 0
        #Upsampling the output of the previous networks
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        #List of networks with growing progressively the output
        self.current_net = nn.ModuleList(
                                []
                                )
        #Network with transforming the output in RGB image
        self.toRGB = nn.ModuleList(
                                []
                                )


        #First linear network
         # get a Latent_dim(100) and transform into w_mult(64) * 128
        first_channels = int(out_resolution_dim *(2**(np.log2(out_resolution_dim)-2)))
        self.gen_fc = nn.Linear(latent_dim, first_channels*4*4, bias=False)
        # get result of Linear layer(8192) transform into (4,4,w_mult(64))
        self.views = PView(first_channels, 4)
        # First batch normalization
        self.first_norm = nn.BatchNorm2d(first_channels)
        # first_norm(x)
        self.activation = nn.ReLU()
        
        #Size of the first iteration

        #Mount the list of networks to doing the progressive processing
        for d in reversed(range(2,int(np.log2(out_resolution_dim)))): # from 2 to number that pow 2 equals the out_resolution_dim
    
            out_channels = int((out_resolution_dim*(2**(d-1) ))/2)
            in_channels = int(out_resolution_dim*(2**(d-1) ))

            self.current_net.append(G_Block(in_channels, out_channels))
            self.toRGB.append(ToRGB(out_channels, channel_num))
   

   

    def forward(self,input):
        #print(input.shape)
        #Make the first input as tensor
        x = self.gen_fc(input)
        #print(x.shape)
        x = self.views(x)
        #print(x.shape)
        x = self.first_norm(x)
        #print(x.shape)
        x = self.activation(x)
        #print(x.shape)


        for block in self.current_net[:self.depth-1]:
            x = block(x)
            #print(x.shape)

        out = self.current_net[self.depth-1](x)
        #print(out.shape)
        x_rgb = self.toRGB[self.depth-1](out)
        #print(x_rgb.shape)

        if self.alpha < 1:
            x_old = self.upsample(x)
            #print(x_old.shape)
            old_rgb = self.toRGB[self.depth-2](x_old)
            #print(old_rgb.shape)
            x_rgb = (1-self.alpha)* old_rgb + self.alpha * x_rgb
            #print(x_rgb.shape)

            self.alpha += self.fade_iters

        return x_rgb

    def growing_net(self, num_iters):
        self.fade_iters = 1/num_iters
        self.alpha = 1/num_iters

        self.depth +=1
