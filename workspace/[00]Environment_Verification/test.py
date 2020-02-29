from __future__ import print_function
import torch

print("Is Cuda available?: "+str(torch.cuda.is_available()))

x = torch.rand(5, 3)
print("random initialized tensor:")
print(x)
