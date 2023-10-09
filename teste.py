import numpy as np

out_resolution_dim = 64

rgb_channels_in = 3
rgb_channels_out = out_resolution_dim

out = out_resolution_dim *(2**(np.log2(out_resolution_dim)-2))
print( "final:", out)

for d in reversed(range(2,int(np.log2(out_resolution_dim)))):
      print( out_resolution_dim*(2**(d-1)))
      
      in_channels = int((out_resolution_dim*(2**(d-1) ))/2)
      out_channels = int(out_resolution_dim*(2**(d-1) ))


      print("RGB in: ", rgb_channels_in, " out: ", int(out_channels/2))
      
      
      print("Conv in: ", in_channels, " out: ", out_channels)


print("Reversed")

rgb_channels_in = 3
rgb_channels_out = out_resolution_dim

for d in reversed(range(2,int(np.log2(out_resolution_dim)))):
      print( out_resolution_dim*(2**(d-1)))
      
      out_channels = int((out_resolution_dim*(2**(d-1) ))/2)
      in_channels = int(out_resolution_dim*(2**(d-1) ))
      
      
      print("Conv in: ", in_channels, " out: ", out_channels)
      print("RGB in: ", out_channels, " out: ", rgb_channels_in)

      


"""      





size_in = 4
#Mount the list of networks to doing the progressive processing
for d in reversed(range(3,int(np.log2(out_resolution_dim)))): # from 2 to number that pow 2 equals the out_resolution_dim
    
    #if d == 2:
    #    in_channels = 370
    #    out_channels = 8
    #    size_in = 2
    #else:
    in_channels = 2**d
    out_channels = 2**(d+1)


    print("in:", in_channels, "x", size_in, "x",size_in, 
          " out:",out_channels, "x", size_in*2, "x", size_in*2)

    size_in *=2


out_resolution_dim = 64
size_out = int(out_resolution_dim/2) 
#Mount the list of networks to doing the progressive processing
for d in range(int(np.log2(out_resolution_dim))-1, 2, -1): # from 2 to number that pow 2 equals the out_resolution_dim
    
    #if d == 2:
    #    in_channels = 370
    #    out_channels = 8
    #    size_in = 2
    #else:
    in_channels = 2**(d+1)
    out_channels = 2**d


    print("in:", in_channels, "x", size_out, "x",size_out, 
          " out:",out_channels, "x", int(size_out/2), "x", int(size_out/2))

    size_out =int(size_out/2)

    """