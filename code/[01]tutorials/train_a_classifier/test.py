import torch
import torchvision # It has data loaders for common datasets such as Imagenet, CIFAR10, MNIST, etc.
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

import numpy as np

# 1. Loading and normalizing CIFAR10

transform = transforms.Compose( # Do multiple transforms in the same time.
    [transforms.ToTensor(), # The output of torchvision datasets are PILImage image. We convert them to Tensors.
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # The images is in range [0,1]. We transform them to range [-1,1]. Normalize(ori_mean,ori_std)

#Download testing data and load them.
testset = torchvision.datasets.CIFAR10(root='/workspace/data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=True, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 2. Define a Convolutional Neural Network

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) # 3 input image channel, 6 output channels, 5x5 square convolution
        self.pool = nn.MaxPool2d(2, 2) # Max pooling over a (2, 2) window
        self.conv2 = nn.Conv2d(6, 16, 5) # 6 input image channel, 16 output channels, 5x5 square convolution
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # img size: 32*32 -> 28*28 -> 14*14
        x = self.pool(F.relu(self.conv2(x))) # img size: 14*14 -> 10*10 -> 5*5
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
print(net)

PATH = './cifar_net.pth'
net.load_state_dict(torch.load(PATH)) # load back

# 3. Inference

dataiter = iter(testloader)
images, labels = dataiter.next() # 4 images
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
outputs = net(images) # ex: [0 0.9 0.1 0 0 0 0 0 0 0] ... * 4

_, predicted = torch.max(outputs, 1)
# max(arr, 1): return max value and its corresponding index 'each row'.
#[0 0.9 0.1 0 0 0 0 0 0 0] -> (0.9 ,1)
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))


# 4. Display images

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0))) # first channel change to zero channel (rgb->gbr)
    plt.show()

# show images
imshow(torchvision.utils.make_grid(images))

while True:
    print("Press 'n' to test next batch. Other input would close the progrm.")
    char = input()
    if char == 'n':
        images, labels = dataiter.next()
        print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))
        imshow(torchvision.utils.make_grid(images))
    else:
        break 