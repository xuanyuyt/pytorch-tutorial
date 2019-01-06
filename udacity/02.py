# -*- coding=utf-8 -*-
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pylab as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# ================================================================== #
#                         Table of Contents                          #
# ================================================================== #
# 1. Load and visualize the data            (Line 22 to 87)
# 2. Define a neural network                (Line 89 to 125)
# 3. Train the model                        (Line 127 to 186)
# 4. Evaluate the performance of our
#    trained model on a test dataset!       (Line 188 to 252)


# ===========================load dataset ========================== #
# number of subprocesses to use for data loading
num_workers = 0
# how many samples per batch to load
batch_size = 20
# percentage of training set to use as validation
valid_size = 0.2

# convert data to torch.FloatTensor
transform = transforms.ToTensor()

# choose the training and test datasets
train_data = datasets.MNIST(root='G:/Other_Datasets/MNIST_data', train=True,
                            download=True, transform=transform)
test_data = datasets.MNIST(root='G:/Other_Datasets/MNIST_data', train=False,
                           download=True, transform=transform)

# obtain training indices that will be used for validation
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# prepare data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                           sampler=train_sampler, num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                           sampler=valid_sampler, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                          num_workers=num_workers)
# ================================================================== #

# ========================= visualize data ========================= #
data_iter = iter(train_loader)
images, labels = data_iter.next()
images = images.numpy()

# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(images[idx]), cmap='gray')
    # print out the correct label for each image
    # .item() gets the value contained in a Tensor
    ax.set_title(str(labels[idx].item()))

img = np.squeeze(images[1])

fig = plt.figure(figsize = (12,12))
ax = fig.add_subplot(111)
ax.imshow(img, cmap='gray')
width, height = img.shape
thresh = img.max()/2.5
for x in range(width):
    for y in range(height):
        val = round(img[x][y],2) if img[x][y] !=0 else 0
        ax.annotate(str(val), xy=(y,x),
                    horizontalalignment='center',
                    verticalalignment='center',
                    color='white' if img[x][y]<thresh else 'black')
# ================================================================== #

# ======================= Define the Network ======================= #
# define the NN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        hidden_1 = 512
        hidden_2 = 512
        # linear layer (784 -> hidden_1)
        self.fc1 = nn.Linear(28 * 28, hidden_1)
        # linear layer (hidden_1 -> hidden_2)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        # dropout layer (p=0.2)
        self.dropout = nn.Dropout(0.2)
        # linear layer (hidden_2 -> 10)
        self.fc3 = nn.Linear(hidden_2, 10)

    def forward(self, x):
        # flatten image input
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# initialize the NN
model = Net().to(device)
print(model)
# ================================================================== #

# ================ Specify Loss Function & Optimizer =============== #
# specify loss function (categorical cross-entropy)
criterion = nn.CrossEntropyLoss()
# specify optimizer (stochastic gradient descent) and learning rate = 0.01
optimizer = optim.SGD(model.parameters(), lr=0.01)
# ================================================================== #

# ======================== Train the Network ======================= #
# number of epochs to train the model
n_epochs = 50
# initialize tracker for minimum validation loss
valid_loss_min = np.Inf

for epoch in range(n_epochs):
    # monitor training loss
    train_loss = 0.0
    valid_loss = 0.0
    # =============== #
    # train the model #
    # =============== #
    model.train() # prep model for training
    for datas, labels in train_loader:
        datas, labels = datas.to(device), labels.to(device)
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        outputs = model(datas)
        # calculate the loss
        loss = criterion(outputs, labels)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update running training loss
        train_loss += loss.item() * datas.size(0)

    # ================== #
    # validate the model #
    # ================== #
    model.eval() # prep model for evaluation
    for datas, labels in valid_loader:
        datas, labels = datas.to(device), labels.to(device)
        # forward pass: compute predicted outputs by passing inputs to the model
        outputs = model(datas)
        # calculate the loss
        loss = criterion(outputs, labels)
        # update running validation loss
        valid_loss += loss.item() * datas.size(0)

    # print training/validation statistics
    # calculate average loss over an epoch
    train_loss = train_loss / len(train_loader.dataset)
    valid_loss = valid_loss / len(valid_loader.dataset)
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch + 1,
        train_loss,
        valid_loss
    ))

    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
        torch.save(model.state_dict(), 'model.pth')
        valid_loss_min = valid_loss
# ================================================================== #

# ==================== Test the Trained Network ==================== #
model.load_state_dict(torch.load('model.pth'))
# initialize lists to monitor test loss and accuracy
test_loss = 0.0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

model.eval() # prep model for evaluation

for datas, labels in test_loader:
    datas, labels = datas.to(device), labels.to(device)
    # forward pass: compute predicted outputs by passing inputs to the model
    outputs = model(datas)
    # calculate the loss
    loss = criterion(outputs, labels)
    # update test loss
    test_loss += loss.item() * datas.size(0)
    # convert output probabilities to predicted class
    _, pred = torch.max(outputs, 1)
    # compare predictions to true label
    correct = np.squeeze(pred.eq(labels.data.view_as(pred)))
    # calculate test accuracy for each object class
    for i in range(batch_size):
        class_correct[labels.data[i]] += correct[i].item()
        class_total[labels.data[i]] += 1

# calculate and print avg test loss
test_loss = test_loss/len(test_loader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(10):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            str(i), 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (i))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))
# ================================================================== #

# ================= Visualize Sample Test Results ================== #
# obtain one batch of test images
data_iter = iter(test_loader)
images, labels = data_iter.next()
images, labels = images.to(device), labels.to(device)

# get sample output
outputs = model(images)
# convert output probabilities to predicted class
_, preds = torch.max(outputs, 1)
# prep images for display
images = images.to('cpu').numpy()


# plot the images in the batch, along with predicted and true labels
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(images[idx]), cmap='gray')
    ax.set_title("{} ({})".format(str(preds[idx].item()), str(labels[idx].item())),
                 color=("green" if preds[idx]==labels[idx] else "red"))
# ================================================================== #
plt.show()