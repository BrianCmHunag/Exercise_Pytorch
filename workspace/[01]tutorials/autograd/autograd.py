import torch
import numpy

#part 1: backward example
print("-----part 1: backward example")
a = numpy.array([[1.0 ,2.0 ],[3.0 ,4.0]])
x = torch.from_numpy(a) # change numpy array to Tensor
x.requires_grad = True #starts to track all operations on it.
y = x + 2
z = y * y * 3
out = z.mean()
#Each tensor has a .grad_fn attribute that references "a Function that has created the Tensor"
print(x)
print(y)
print(z)
print(out)

print("x, y, z, out requires_grad?")
print (x.requires_grad)
print (y.requires_grad)
print (z.requires_grad)
print (out.requires_grad)

out.backward() # dLoss/dout * dout/dz * dz/dy * dy/dx. 'dLoss/dout' is set to 1 by default if 'out' is a scalar.
print(x.grad)
# dLoss/dout = 1
# dout/dz = 1 (mean)
# dz/dy= 3y/2
# dy/dx =1
# 1*1*3y/2*1 = 3(x+2)/2 : x=1 -> 4.5 , 2->9 ...

# part 2: loss is not a scalar
print("-----part 2: loss is not a scalar")
x = torch.ones(2, 2, requires_grad=True) #starts to track all operations on it.
y = (x+1)*(x+1)

print(x)
print(y)
v = torch.tensor([[2.0, 1.0],[2.0, 1.0]], dtype=torch.float) # y is not a scalar, so we need to assign a vector as the gradient of a scalar function (dLoss/dy).
y.backward(v)
print(x.grad)

# part 3: no requre gradient
print("-----part 3: no requre gradient")
x = torch.ones(2, 2, requires_grad=True)
with torch.no_grad():
    y = x * x

print("x, y requires_grad?")
print (x.requires_grad)
print (y.requires_grad)

print(x)
print(y)
try:
    v = torch.tensor([[2.0, 1.0],[2.0, 1.0]], dtype=torch.float) # y is not a scalar, so we need to assign a vector as the gradient of a scalar function(dLoss/dy).
    y.backward(v)
    print(x.grad)
except:
    print("Can not backward.")
