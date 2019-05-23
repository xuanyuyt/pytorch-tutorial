import torch
import numpy as np
# ================================================================== #
#                             1. Tensor                              #
# ================================================================== #
# Construct a matrix filled zeros and of dtype long:
x = torch.empty((5, 3)) # -> np.empty((5, 3))
print(x)

# Construct a randomly initialized matrix:
x = torch.rand(5, 3) # -> np.random.rand(5, 3)
print(x)

# Construct a matrix filled zeros and of dtype long:
x = torch.zeros((5, 3), dtype=torch.long) # -> x = np.zeros((5, 3), dtype = 'int64')
print(x)

# Construct a randomly initialized matrix:
x = torch.tensor([5.5, 3])
print(x)

# or create a tensor based on an existing tensor.
x = x.new_ones((5, 3), dtype=torch.double)      # new_* methods take in sizes
print(x)
x = torch.randn_like(x, dtype=torch.float)    # override dtype!
print(x)
print(x.size())

# Operations
y = torch.rand(5, 3)
print(x + y)
print(torch.add(x, y))

result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)

y.add_(x)
print(y)

# Resizing: If you want to resize/reshape tensor, you can use Tensor.view or Tensor.reshape:
x = torch.randn(4, 4)
y = x.view(16) # x.reshape(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())

# If you have a one element tensor, use .item() to get the value as a Python number
x = torch.randn(1)
print(x)
print(x.item())

# The Torch Tensor and NumPy array will share their underlying memory locations
# (if the Torch Tensor is on CPU), and changing one will change the other.
# Convert the torch tensor to a numpy array.
a = torch.ones(5)
b = a.numpy()
a.add_(1)
print(a)
print(b)

# Convert the numpy array to a torch tensor.
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)


# let us run this cell only if CUDA is available
# We will use ``torch.device`` objects to move tensors in and out of GPU
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!