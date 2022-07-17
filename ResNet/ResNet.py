import torch
import torch.nn as nn

class block(nn.Module):
    def __init__(self, in_channels, intermediate_channels, identity_downsample=None, stride=1): #identity downsample is a conv layer that we need to do if we change the input size or number of channels
        super(block,self).__init__()
        self.expansion = 4 #end of the block it needs to be 4 times
        self.conv1 = nn.Conv2d(in_channels, intermediate_channels, kernel_size=1, stride=1, padding=0 ) #eg 1x1 , 64 : here we also crt the output channels 256 to 128
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.conv2 = nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=3, stride=stride, padding=1 ) #eg 3x3, 64  # here stride is diffrent to decrease the output x*y to eg from  56x56 to 28x28 (but only in the first run of a layer)
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.conv3 = nn.Conv2d(intermediate_channels, intermediate_channels*self.expansion, kernel_size=1, stride=1, padding=0) # 1x1, 256  : here we increse output channels eg 128 to 512
        self.bn3 = nn.BatchNorm2d(intermediate_channels*self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample #conv layer so that its in same shape  later on in the layers
        self.stride = stride

    def forward(self,x):
        identity = x.clone()
        print('start')
        print(x.shape)
        x = self.conv1(x)
        
        x = self.bn1(x)
        x = self.relu(x)
        print('conv1')
        print(x.shape)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        print('conv2')
        print(x.shape)
        x = self.conv3(x)
        x = self.bn3(x)
        print('conv3')
        print(x.shape)
        # now add identiy before relu
        # condition to put before identiy # if we need to change shape
        #only for the first block
        if self.identity_downsample is not None:
            print('identity_downsampled')
            print(identity.shape)
            identity = self.identity_downsample(identity)
            print(identity.shape)

        x += identity
        x = self.relu(x)
        return x

class ResNet(nn.Module): # layers will tell how many time s to use block eg 3 = layer[0] time hence layer = [3, 4, 6, 3]
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels,64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        #  ResNet layers
        self.layers1 = self._make_layer(block, layers[0], intermediate_channels=64, stride = 1)
        self.layers2 = self._make_layer(block, layers[1], intermediate_channels=128, stride = 2)
        self.layers3 = self._make_layer(block, layers[2], intermediate_channels=256, stride = 2)
        self.layers4 = self._make_layer(block, layers[3], intermediate_channels=512, stride = 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1)) #fixes the input to desired output size
        self.fc = nn.Linear(512*4, num_classes)


    def forward(self,x):
        #print(x.shape)
        x = self.conv1(x) # output  torch.Size([4, 64, 112, 112])
        x = self.bn1(x)
        x = self.relu(x)
        #print(x.shape)
        x = self.maxpool(x) #torch.Size([4, 64, 56, 56])
        #print(x.shape)
        print("layer 1")
        x = self.layers1(x)
        print("layer 2")
        x = self.layers2(x)
        print("layer 3")
        x = self.layers3(x)
        print("layer 4")
        x = self.layers4(x)
        print("avgpool")
        x = self.avgpool(x)
        print(x.shape) # to make shure its crt shape, output (1,1)
        x = x.reshape(x.shape[0],-1)
        print(x.shape)
        x = self.fc(x)
        return x



    def _make_layer(self, block, num_residual_blocks, intermediate_channels, stride ):

        # num of intermediate_channels at the end of the layer
        # so stride because one of the block si going to have a stride of 2
        identity_downsample = None
        layers = []
        # there is changing of identiy when the stride is not 1 or ...
        if stride !=1 or self.in_channels != intermediate_channels *4:  #we crt no of out channels and x*y shape if necessary of the identity
            identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels,intermediate_channels*4, kernel_size=1, stride=stride),
                                   nn.BatchNorm2d(intermediate_channels*4))

        layers.append(block(self.in_channels, intermediate_channels, identity_downsample,stride)) #so now the output would be 64(but inside the block it becomes 256 and comes out), but we need to run the block 3 times
        self.in_channels = intermediate_channels*4 #so output will be 256 though u give 64 as ur output cahnnels
        

        for i in range(num_residual_blocks -1): #because w ehave done 1 of it top
            layers.append(block(self.in_channels, intermediate_channels)) # in channel here will be 256 and out inside the block will become 64 and then 64*4 = 256 back again
            #remeber here stride is 1 and we dont use identity downsample here

        return nn.Sequential(*layers)
           
def ResNet50(img_channels=3, num_classes=1000):
    return ResNet(block, [3, 4, 6, 3], img_channels, num_classes)

def ResNet101(img_channels=3, num_classes=1000):
    return ResNet(block, [3, 4, 23, 3], img_channels, num_classes)

def ResNet150(img_channels=3, num_classes=1000):
    return ResNet(block, [3, 8, 36, 3], img_channels, num_classes)

def test():
    net = ResNet50(img_channels=3, num_classes=1000)
    y = net(torch.randn(4, 3, 224, 224)).to("cuda")
    print(y.size())

test()