# License: BSD
# Author: Ghassen Hamrouni

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from torch.utils.data import Dataset
import elasticdeform
import cv2

plt.ion()   # interactive mode

from six.moves import urllib
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        return x, y

    def __len__(self):
        return len(self.data)

# Code from: https://stackoverflow.com/questions/37119071/scipy-rotate-and-zoom-an-image-without-changing-its-dimensions
def cv2_clipped_zoom(img, zoom_factor=0):
    """
    Center zoom in/out of the given image and returning an enlarged/shrinked view of
    the image without changing dimensions
    ------
    Args:
        img : ndarray
            Image array
        zoom_factor : float
            amount of zoom as a ratio [0 to Inf). Default 0.
    ------
    Returns:
        result: ndarray
           numpy ndarray of the same shape of the input img zoomed by the specified factor.
    """
    if zoom_factor == 0:
        return img

    height, width = img.shape[:2]  # It's also the final desired shape
    new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)

    ### Crop only the part that will remain in the result (more efficient)
    # Centered bbox of the final desired size in resized (larger/smaller) image coordinates
    y1, x1 = max(0, new_height - height) // 2, max(0, new_width - width) // 2
    y2, x2 = y1 + height, x1 + width
    bbox = np.array([y1, x1, y2, x2])
    # Map back to original image coordinates
    bbox = (bbox / zoom_factor).astype(np.int)
    y1, x1, y2, x2 = bbox
    cropped_img = img[y1:y2, x1:x2]

    # Handle padding when downscaling
    resize_height, resize_width = min(new_height, height), min(new_width, width)
    pad_height1, pad_width1 = (height - resize_height) // 2, (width - resize_width) // 2
    pad_height2, pad_width2 = (height - resize_height) - pad_height1, (width - resize_width) - pad_width1
    pad_spec = [(pad_height1, pad_height2), (pad_width1, pad_width2)] + [(0, 0)] * (img.ndim - 2)

    result = cv2.resize(cropped_img, (resize_width, resize_height))
    result = np.pad(result, pad_spec, mode='constant')
    assert result.shape[0] == height and result.shape[1] == width
    return result

def get_data():
    trainset = datasets.MNIST('./data', download=True, train=True)
    testset = datasets.MNIST('./data', download=True, train=False)

    return trainset, testset

def get_deformed_dataloader(batch_size):
    img_index = 0
    label_index = 1
    trainset, testset = get_data()

    customTrainDataset = CustomDataset([], [])
    customTestDataSet = CustomDataset([], [])
    convert_tensor = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                   ])

    for data in trainset:
        # convert PIL to np array
        image = np.asarray(data[img_index])
        label = data[label_index]

        # convert image to tensor
        tensor_image = convert_tensor(image)

        # randomly deform data
        random_tensor = convert_tensor(elasticdeform.deform_random_grid(image, sigma=25, points=3))

        # zoom deform
        displacement = np.full((2, 3, 3), 0)
        zoomed_image = elasticdeform.deform_grid(image, displacement,
                                               prefilter=False, zoom=.25)
        zoomed_tensor = convert_tensor(zoomed_image)

        # add original image to dataset
        customTrainDataset.data.append(deepcopy(tensor_image))
        customTrainDataset.targets.append(deepcopy(label))

        # add randomly mutated image to dataset
        customTrainDataset.data.append(deepcopy(random_tensor))
        customTrainDataset.targets.append(deepcopy(label))

        #add zoom deformed image to dataset
        customTrainDataset.data.append(deepcopy(zoomed_tensor))
        customTrainDataset.targets.append(deepcopy(label))

    for data in testset:
        # convert PIL to np array
        image = np.asarray(data[img_index])
        label = data[label_index]

        # convert image to tensor
        tensor_image = convert_tensor(image)

        # randomly deform data
        random_tensor = convert_tensor(elasticdeform.deform_random_grid(image, sigma=25, points=3))

        # zoom deform
        displacement = np.full((2, 3, 3), 0)
        zoomed_image = elasticdeform.deform_grid(image, displacement,
                                               prefilter=False, zoom=.25)
        zoomed_tensor = convert_tensor(zoomed_image)

        # add original image to dataset
        customTestDataSet.data.append(deepcopy(tensor_image))
        customTestDataSet.targets.append(deepcopy(label))

        # add randomly mutated image to dataset
        customTestDataSet.data.append(deepcopy(random_tensor))
        customTestDataSet.targets.append(deepcopy(label))

        #add zoom deformed image to dataset
        customTestDataSet.data.append(deepcopy(zoomed_tensor))
        customTestDataSet.targets.append(deepcopy(label))


    trainloader = torch.utils.data.DataLoader(customTrainDataset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(customTestDataSet, batch_size=batch_size, shuffle=True)

    return trainloader, testloader


train_loader, test_loader = get_deformed_dataloader(64)

# # Training dataset
# train_loader, = torch.utils.data.DataLoader(
#     datasets.MNIST(root='.', train=True, download=True,
#                    transform=transforms.Compose([
#                        transforms.ToTensor(),
#                        transforms.Normalize((0.1307,), (0.3081,))
#                    ])), batch_size=64, shuffle=True, num_workers=4)
# # Test dataset
# test_loader = torch.utils.data.DataLoader(
#     datasets.MNIST(root='.', train=False, transform=transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.1307,), (0.3081,))
#     ])), batch_size=64, shuffle=True, num_workers=4)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)

        # Perform the usual forward pass
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


model = Net().to(device)

optimizer = optim.SGD(model.parameters(), lr=0.01)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 500 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    return loss.item()
#
# A simple test procedure to measure the STN performances on MNIST.
#


def test():
    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # sum up batch loss
            test_loss += F.nll_loss(output, target, size_average=False).item()
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
              .format(test_loss, correct, len(test_loader.dataset),
                      100. * correct / len(test_loader.dataset)))

    return test_loss, 100. * correct / len(test_loader.dataset)

def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp

# We want to visualize the output of the spatial transformers layer
# after the training, we visualize a batch of input images and
# the corresponding transformed batch using STN.


def visualize_stn():
    with torch.no_grad():
        # Get a batch of training data
        data = next(iter(test_loader))[0].to(device)

        input_tensor = data.cpu()
        transformed_input_tensor = model.stn(data).cpu()

        in_grid = convert_image_np(
            torchvision.utils.make_grid(input_tensor))

        out_grid = convert_image_np(
            torchvision.utils.make_grid(transformed_input_tensor))

        # Plot the results side-by-side
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(in_grid)
        axarr[0].set_title('Dataset Images')

        axarr[1].imshow(out_grid)
        axarr[1].set_title('Transformed Images')

train_error_record = []
test_error_record = []
accuracy_record = []

for epoch in range(1, 30 + 1):
    train_error = train(epoch)
    test_error, accuracy = test()

    train_error_record.append(train_error)
    test_error_record.append(test_error)
    accuracy_record.append(accuracy)

print(train_error_record)
print(test_error_record)
print(accuracy_record)

# Visualize the STN transformation on some input batch
visualize_stn()

plt.ioff()
plt.show()
