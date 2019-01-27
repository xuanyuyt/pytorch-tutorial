# -*- coding:utf-8 -*-
import torch
import torchvision
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt

def imshow(img):
    img = img / 2 + 0.5 # unnormalize
    plt.imshow(np.transpose(img, (1,2,0))) # convert form Tesor image(CHW->HWC)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

########################################################################
# 1. Load and normalizing the CIFAR10 training and test datasets
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1].

# how many samples per batch to load
batch_size = 40
# percentage of training set to use as validation
valid_size = 0.2

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(), # randomly flip and rotate
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# choose the training and testing datasets
train_data = datasets.CIFAR10('G:/Other_Datasets/cifar', train=True,
                              transform=transform, download=True)

test_data = datasets.CIFAR10('G:/Other_Datasets/cifar', train=False,
                             transform=transform, download=True)

# obtain training indices that will be used for validation
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# prepare data loader(combine dataset and sampler)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False,
                                           sampler=train_sampler, num_workers=0)
valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False,
                                           sampler=valid_sampler, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False,
                                          num_workers=0)

# specify the image classes
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()

# show iamges
imshow(torchvision.utils.make_grid(images))

# orint labels
print(''.join('%6s' % classes[labels[j]] for j in range(batch_size)))
print(images[0].shape)
plt.show()

########################################################################
# 2. Define a Convolutional Neural Network
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=64*4*4, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=10)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64*4*4)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

model = Net()
print(model)

########################################################################
# 3. Define a Loss function and optimizer
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Let's use a Classification Cross-Entropy loss and SGD with momentum.

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

########################################################################
# 4. Train the network
# ^^^^^^^^^^^^^^^^^^^^
# This is when things start to get interesting.
# We simply have to loop over our data iterator, and feed the inputs to the
# network and optimize.
model = model.to(device)

n_epochs = 8
valid_loss_min = np.Inf

for epoch in range(n_epochs): # loop over the dataset mutiple times
    ###################
    # train the model #
    ###################
    train_loss = 0.0
    model.train()
    for datas, labels in train_loader:
        # get the input
        datas, labels = datas.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(datas)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * datas.size(0)

    ######################
    # validate the model #
    ######################
    valid_loss = 0.0
    model.eval()
    for datas, labels in valid_loader:
        # get the input
        datas, labels = datas.to(device), labels.to(device)

        # forward
        outputs = model(datas)
        loss = criterion(outputs, labels)

        valid_loss += loss.item() * datas.size(0)

    # print statistics
    train_loss = train_loss / len(train_loader.dataset)
    valid_loss = valid_loss / len(valid_loader.dataset)
    print("Eopch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:6f}".format(
        epoch + 1,
        train_loss,
        valid_loss
    ))

    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
        torch.save(model.state_dict(), 'model_cifar.pth')
        valid_loss_min = valid_loss

print('Finished Training')

########################################################################
# 5. Test the network on the test data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

model.load_state_dict(torch.load('model_cifar.pth'))

test_loss = 0.0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
model.eval()

for images, labels in test_loader:
    # get the input
    datas, labels = images.to(device), labels.to(device)

    # forward
    outputs = model(datas)
    loss = criterion(outputs, labels)

    test_loss += loss.item() * datas.size(0)
    # convert output probabilities to predicted class
    _, preds = torch.max(outputs, 1)
    correct = (preds == labels).squeeze()

    for i in range(batch_size):
        label = labels[i].item()
        class_correct[label] += correct[i].item()
        class_total[label] += 1

# average test loss
test_loss = test_loss/len(test_loader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(10):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            classes[i], 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))


# Visualize Sample Test Results
dataiter = iter(test_loader)
images, labels = dataiter.next()
datas, labels = images.to(device), labels.to(device)
outputs = model(datas)
_, preds = torch.max(outputs, 1)
# plot the images in the batch, along with predicted and true labels
fig = plt.figure(figsize=(14, 8))
for idx in np.arange(batch_size):
    ax = fig.add_subplot(4, batch_size/4, idx+1, xticks=[], yticks=[])
    img = images[idx] / 2 + 0.5  # unnormalize
    ax.imshow(np.transpose(img.numpy(), (1, 2, 0)))
    ax.set_title("{} ({})".format(classes[preds[idx]], classes[labels[idx]]),
                 color=("green" if preds[idx] == labels[idx].item() else "red"))
plt.show()
del dataiter