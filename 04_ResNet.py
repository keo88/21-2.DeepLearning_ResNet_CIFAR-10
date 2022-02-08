import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np
import time

########################################
# You can define whatever classes if needed
########################################


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_pooling_block=False):
        super(ResBlock, self).__init__()

        if is_pooling_block:
            stride_val = 2
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)
        else:
            stride_val = 1
            self.shortcut = nn.Identity()

        self.residual = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride_val, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        out = self.residual(x) + self.shortcut(x)
        return out


class ResStage(nn.Module):
    def __init__(self, nblk, in_channels, out_channels, is_pooling_stage):
        super(ResStage, self).__init__()
        self.blocks = nn.Sequential(
            ResBlock(in_channels, out_channels, is_pooling_block=is_pooling_stage)
        )
        for blk in range(nblk - 1):
            self.blocks.add_module('blk-' + str(blk), ResBlock(out_channels, out_channels, False))

    def forward(self, x):
        out = self.blocks(x)
        return out


class IdentityResNet(nn.Module):
    
    # __init__ takes 4 parameters
    # nblk_stage1: number of blocks in stage 1, nblk_stage2.. similar
    def __init__(self, nblk_stage1, nblk_stage2, nblk_stage3, nblk_stage4):
        super(IdentityResNet, self).__init__()
    ########################################
    # Implement the network
    # You can declare whatever variables
    ########################################
        self.resnetConv = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            ResStage(nblk_stage1, 64, 64, False),
            ResStage(nblk_stage2, 64, 128, True),
            ResStage(nblk_stage3, 128, 256, True),
            ResStage(nblk_stage4, 256, 512 , True),
        )
        self.fc = nn.Linear(512, 10)
    ########################################
    # You can define whatever methods
    ########################################
    
    def forward(self, x):
        conv_result = F.avg_pool2d(self.resnetConv(x), kernel_size=4, stride=4)
        out = self.fc(torch.reshape(conv_result, (conv_result.shape[0], -1)))
        return out


########################################
# Q1. set device
# First, check availability of GPU.
# If available, set dev to "cuda:0";
# otherwise set dev to "cpu"
########################################


dev = 'cuda:0'
print('current device: ', dev)

########################################
# data preparation: CIFAR10
########################################

########################################
# Q2. set batch size
# set batch size for training data
########################################
batch_size = 4

# preprocessing
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# load training data
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)

# load test data
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# define network
net = IdentityResNet(nblk_stage1=2, nblk_stage2=2,
                     nblk_stage3=2, nblk_stage4=2)

########################################
# Q3. load model to GPU
# Complete below to load model to GPU
########################################
net = net.to(dev)

# set loss function
criterion = nn.CrossEntropyLoss()

########################################
# Q4. optimizer
# Complete below to use SGD with momentum (alpha= 0.9)
# set proper learning rate
########################################
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

# start training
t_start = time.time()

for epoch in range(5):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(dev), data[1].to(dev)

        ########################################
        # Q5. make sure gradients are zero!
        # zero the parameter gradients
        ########################################
        optimizer.zero_grad()

        ########################################
        # Q6. perform forward pass
        ########################################
        outputs = net(inputs)

        # set loss
        loss = criterion(outputs, labels)

        ########################################
        # Q7. perform backprop
        ########################################
        loss.backward()

        ########################################
        # Q8. take a SGD step
        ########################################
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
            t_end = time.time()
            print('elapsed:', t_end - t_start, ' sec')
            t_start = t_end

print('Finished Training')

# now testing
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

########################################
# Q9. complete below
# when testing, computation is done without building graphs
########################################
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(dev), data[1].to(dev)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

# per-class accuracy
for i in range(10):
    print('Accuracy of %5s' % (classes[i]), ': ',
          100 * class_correct[i] / class_total[i], '%')

# overall accuracy
print('Overall Accurracy: ', (sum(class_correct) / sum(class_total)) * 100, '%')