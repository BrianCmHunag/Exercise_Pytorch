import torch
import torchvision # It has data loaders for common datasets such as Imagenet, CIFAR10, MNIST, etc.
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

# 1. Loading and normalizing CIFAR10

transform = transforms.Compose( # Do multiple transforms in the same time.
    [transforms.ToTensor(), # The output of torchvision datasets are PILImage image. We convert them to Tensors.
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # The images is in range [0,1]. We transform them to range [-1,1]. Normalize(ori_mean,ori_std)

#Download trainning data and load them.
trainset = torchvision.datasets.CIFAR10(root='/workspace/data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
#batch_size: how many samples per batch to load
#num_workers: Set it as a positive integer will turn on multi-process data loading with the specified number of loader worker processes.

#Download testing data and load them.
testset = torchvision.datasets.CIFAR10(root='/workspace/data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

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

# 3. Define a Loss function and optimizer

criterion = nn.CrossEntropyLoss() #good for classification
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) # The learnable parameters of a model are returned by net.parameters()

# 4. Train the networkfor epoch in range(2), then save the model.

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    # enumerate() return {index, data} ex: [a,b]-> {0, 'a'},{1, 'b'}. Second parameter is enumerate start index.
    # Due to batch size=4, trainloader only has 12500 image sets. Each set has 4 images.
    for i, data in enumerate(trainloader, 0): # i = 0~12499
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()
        # Zero the parameter gradients, not zero the weights and bias.
        # Because PyTorch accumulates the gradients on subsequent backward passes, use zero_grad() to get correct gradients.

        # forward + backward + optimize (4 images per update)
        outputs = net(inputs) #forward
        loss = criterion(outputs, labels) #loss CrossEntropyLoss need: output [0.2,0.7,0.1,0,...] vs labels [1]
        loss.backward() #backward
        optimizer.step() #update weights and bias

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

# 5. Load the model and then test

net = Net()
net.load_state_dict(torch.load(PATH)) # load back

dataiter = iter(testloader)
images, labels = dataiter.next() # 4 images
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
outputs = net(images) # ex: [0 0.9 0.1 0 0 0 0 0 0 0] ... * 4

_, predicted = torch.max(outputs, 1)
# max(arr, 1): return max value and its corresponding index 'each row'.
#[0 0.9 0.1 0 0 0 0 0 0 0] -> (0.9 ,1)
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

#Perform on the whole dataset.
correct = 0
total = 0
with torch.no_grad(): # inference, so we don't require the gradients.
    for data in testloader: #0~2499
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1) # from one hot to index
        total += labels.size(0) # row size = 4
        correct += (predicted == labels).sum().item() #correct number 0~4

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

# See the result for each class

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1) # from one hot to index
        c = (predicted == labels).squeeze() # make array be more clear
        for i in range(4):
            label = labels[i] # answer label
            class_correct[label] += c[i].item() #c[i]= flase or true
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
