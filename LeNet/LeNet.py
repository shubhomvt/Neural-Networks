from multiprocessing import pool
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision.datasets as datasets
import torchvision.transforms as transform

#LeNet architecure
#1x32x32 -> (5x5),s=1,p=0 -> avg pool s=2,p=0 -> (5x5), s=1,p=0 ->
# -> Conv 5x5 to 120 x Linear 84 x Linear 10

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(kernel_size=(2,2), stride = (2,2))
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6,kernel_size=(5,5),stride=(1,1),padding=(2,2))
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5,5), stride=(1,1),padding=(0,0))
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=(5,5), stride=(1,1),padding=(0,0))
        self.linear1 = nn.Linear(120,84)
        self.linear2 = nn.Linear(84, 10 )

    def forward(self,x):
        x=self.relu(self.conv1(x))
        x=self.pool(x)
        x=self.relu(self.conv2(x))
        x=self.pool(x)
        x=self.relu(self.conv3(x))# num_examples x 120 x 1 x 1 -> num_examples x 120
        x = x.reshape(x.shape[0],-1)
        x=self.relu(self.linear1(x))
        x=self.linear2(x)
        return x
 
# model = LeNet()
# x = torch.randn(64,1,32,32)

# print(model(x).shape)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )

#hyperparameter

in_channels = 1
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 1

#load data

train_dataset = datasets.MNIST(root='dataset/', train=True, transform= transform.ToTensor(),download = True) #where it should store the dataset..folder name dataset
train_loader = DataLoader(dataset=train_dataset , batch_size=batch_size,shuffle=True)#shuffle the batches after every epoch 

test_dataset = datasets.MNIST(root='dataset/', train=False, transform= transform.ToTensor(),download = True) #where it should store the dataset..folder name dataset
test_loader = DataLoader(dataset=test_dataset , batch_size=batch_size,shuffle=True)#shuffle the batches after every epoch 


#intialize network
model = LeNet().to(device)

#Loss and optimizer
criterion = nn.CrossEntropyLoss() #for our loss function we use this
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


#Train

for epoch in range(num_epochs):
    for batch_idx ,(data,targets) in enumerate(train_loader): #dta for images,targets for soloution
        #get data to cuda if possible
        data = data.to(device=device) #data tensor to device that we are useing
        targets=targets.to(device=device)

        #print(data.shape) #found out that the shape was [64,1,28,28] but we want to unroll it [64 784]
        

        #forward part of NN
        scores = model(data)
        loss = criterion(scores,targets)

        #backward
        optimizer.zero_grad() #set grad to zero after each batch so that it doesnt store data from previous forwrds pass
        loss.backward()

        #gradient descent or adam step
        optimizer.step()

# Check the accuracy on training

def check_accuracy(loader,model):
    num_correct = 0
    num_samples = 0
    # we want to set the model to evalute....
    model.eval()
    
    with torch.no_grad():
        for x, y in loader:
            #same like we did while training
            x=x.to(device=device)
            y=y.to(device=device)

            

            scores = model(x) # op will be shape 64,10
            _,prediction = scores.max(1) #index of max value for sceond dimention
            num_correct += (prediction == y).sum()
            num_samples += prediction.size(0) #first dimention 64

            print(f'Got {num_correct}/{num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')

    model.train() #to check the accuracy while training
    

check_accuracy(train_loader,model)
check_accuracy(test_loader,model)