import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from copy import deepcopy
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms


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


def get_custom_data(size, batch_size):
    img_index = 0
    label_index = 1

    counter = np.zeros(size)
    customDataset = CustomDataset([], [])

    trainset, _ = get_data(batch_size)

    for data in trainset:
        image = data[img_index]
        label = data[label_index]

        if counter[label] <= size:
            customDataset.data.append(deepcopy(image))
            customDataset.targets.append(deepcopy(label))
            counter[label] += 1

        if sum(counter) == size*10:
            break

    trainloader = torch.utils.data.DataLoader(customDataset, batch_size=batch_size, shuffle=True)
    return trainloader

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.show()

def get_data(batch_size):
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ]),
    }

    trainset = datasets.MNIST('./data', download=True, train=True, transform=data_transforms['train'])
    testset = datasets.MNIST('./data', download=True, train=False, transform=data_transforms['val'])

    return trainset, testset

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return loss.item()


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, correct


def visualize_data(dataloader):
    # Get a batch of training data
    inputs, classes = next(iter(dataloader))

    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)

    imshow(out)

def create_plot(x_axis, errors):
    i = 0
    for error in errors:
        i += 10
        graph_name = "size %s*10" % str(i)
        plt.plot(x_axis, error, label=graph_name)

    plt.xlabel("Epochs")
    plt.ylabel("Train Error")
    plt.title("Train Error vs Epochs for ResNet50")
    plt.legend()
    plt.show()

def create_pretrained_resnet():
    model = models.resnet50(pretrained=True)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)

    # freeze layers
    layers = 0
    for child in model.children():
        layers += 1
        if layers < 9:
            for param in child.parameters():
                param.requires_grad = False

    return model

def create_pretrained_vgg():
    model = models.vgg19(pretrained=True)
    model.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, 10)

    # freeze layers
    for child in model.features:
        for param in child.parameters():
            param.requires_grad = False

    layers = 0
    for child in model.classifier:
        layers += 1
        if layers < 6:
            for param in child.parameters():
                param.requires_grad = False

    return model

def main():
    epochs = 50
    x_axis = np.arange(1, epochs + 1, step=1)
    sizes = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    batch_size = 10
    train_error_record = []

    for size in sizes:
        model = create_pretrained_resnet()
        # model = create_pretrained_vgg()
        criterion = nn.CrossEntropyLoss()
        optimizer_ft = optim.SGD(model.parameters(), lr=0.01)

        train_errors = []
        trainloader = get_custom_data(size, batch_size)
        for epoch in range(0, epochs):
            print("Epoch %s:" % epoch)
            train_loss = train(trainloader, model, criterion, optimizer_ft)
            train_errors.append(train_loss)
        train_error_record.append(train_errors)

    print(train_error_record)
    create_plot(x_axis, train_error_record)


if __name__ == '__main__':
    main()
