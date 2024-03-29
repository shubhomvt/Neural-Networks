import torch
import torch.nn as nn




architecture_config=[
    #tuple is (kernel_size,num_filter,stride,padding)
    (7,64,2,3),
    "M",
    (3,192,1,1),
    "M",
    (1,128,1,0),
    (3,256,1,1),
    (1,256,1,1),
    (3,512,1,1),
    "M",
    #list: tuples and the last integer represents nummber of repeats
    [(1,256,1,0),(3,512,1,1),4],
    (1,512,1,0),
    (3,1024,1,1),
    "M",
    [(1,512,1,0),(3,1024,1,1),2],
    (3,1024,1,1),
    (3,1024,2,1),
    (3,1024,1,1),
    (3,1024,1,1)



]

class CNNBlock(nn.Module):  #use this multiple time
    def __init__(self, in_channels, out_channels,**kwargs ) :   #for key word arguments ->set **kwargs
        super(CNNBlock,self).__init__()      #https://stackoverflow.com/questions/576169/understanding-python-super-with-init-methods       #initilize the cnnblock     #use of super indirection (ability to reference base object with super())
        self.conv = nn.Conv2d(in_channels,out_channels, bias=False, **kwargs)
        self.batchnorm = nn.batchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyRelu(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))

        