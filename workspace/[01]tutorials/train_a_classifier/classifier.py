import torch
import torchvision # It has data loaders for common datasets such as Imagenet, CIFAR10, MNIST, etc.
import torchvision.transforms as transforms

# 1. Loading and normalizing CIFAR10

transform = transforms.Compose( # Do multiple transforms in the same time.
    [transforms.ToTensor(), # The output of torchvision datasets are PILImage image. We convert them to Tensors.
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # The images is in range [0,1]. We transform them to range [-1,1]. Normalize(ori_mean,ori_std)

#Download trainning data and load them.
trainset = torchvision.datasets.CIFAR10(root='~/program/data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
#Set num_workers as a positive integer will turn on multi-process data loading with the specified number of loader worker processes.

#Download testing data and load them.
testset = torchvision.datasets.CIFAR10(root='~/program/data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
