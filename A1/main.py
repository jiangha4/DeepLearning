import torch
import customDataset
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# Define neural network class
# Base Class for network
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# One Layer NN
class OneLayerNN(Network):
    def __init__(self):
        super(OneLayerNN, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 1024),
            nn.ReLU(),
            nn.Linear(1024, 10)
        )


# Two Layer NN
class TwoLayerNN(Network):
    def __init__(self):
        super(TwoLayerNN, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 10)
        )


# Two Layer NN using Sigmoid
class TwoLayerNNSigmoid(Network):
    def __init__(self):
        super(TwoLayerNNSigmoid, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 1024),
            nn.Sigmoid(),
            nn.Linear(1024, 1024),
            nn.Sigmoid(),
            nn.Linear(1024, 10)
        )


def train_loop(dataloader, model, loss_fn, optimizer):
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


def test_loop(dataloader, model, loss_fn):
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


def get_dataset():
    trainset = datasets.FashionMNIST('./data',
                                     download=True,
                                     train=True,
                                     transform=transforms.ToTensor())

    testset = datasets.FashionMNIST('./data',
                                    download=True,
                                    train=False,
                                    transform=transforms.ToTensor())

    return trainset, testset


def get_data(batch_size):
    trainset, testset = get_dataset()

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False)

    return trainloader, testloader


def get_polluted_data(batch_size, pollutationRate):
    trainset, testset = get_dataset()

    polluted_trainset = customDataset.create_polluted_data(trainset, pollutationRate)

    trainloader = torch.utils.data.DataLoader(polluted_trainset, batch_size=batch_size,
                                              shuffle=True)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False)

    return trainloader, testloader


def get_rotated_data(batch_size, shift, dim):
    trainset, testset = get_dataset()

    rotated_testset = customDataset.rotate_dataset(testset, shift, dim)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True)
    testloader = torch.utils.data.DataLoader(rotated_testset, batch_size=batch_size,
                                             shuffle=False)

    return trainloader, testloader


def question_two():
    learning_rate = 0.001
    momentum = 0
    batch_size = 30
    epochs = 2

    train_loader, test_loader = get_data(batch_size)

    one_layer_model = OneLayerNN()
    two_layer_model = TwoLayerNN()
    loss_fn = nn.CrossEntropyLoss()
    one_layer_optimizer = torch.optim.SGD(one_layer_model.parameters(), lr=learning_rate, momentum=momentum)
    two_layer_optimizer = torch.optim.SGD(two_layer_model.parameters(), lr=learning_rate, momentum=momentum)

    # Run experiment for one layer NN
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_loader, one_layer_model, loss_fn, one_layer_optimizer)
        test_loop(test_loader, one_layer_model, loss_fn)
        print("Done!")

    # Run experiment for two layer NN
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_loader, two_layer_model, loss_fn, two_layer_optimizer)
        test_loop(test_loader, two_layer_model, loss_fn)
        print("Done!")


def question_three():
    # Question 3, part a: Mini-batch size of 1, 10, 1000
    batch_sizes = [1, 10, 1000]
    # Question 3, part b: Vary the learning rate as 1.0, 0.1, 0.01, 0.001
    learning_rates = [0.1, 0.01, 0.001]
    momentum = 0
    epochs = 2

    for batch_size in batch_sizes:
        train_loader, test_loader = get_data(batch_size)

        for learning_rate in learning_rates:
            loss_fn_relu = nn.CrossEntropyLoss()
            loss_fn_sigmoid = nn.CrossEntropyLoss()

            two_layer_relu_model = TwoLayerNN()
            two_layer_sigmoid_model = TwoLayerNNSigmoid()

            optimizer_relu = torch.optim.SGD(two_layer_relu_model.parameters(), lr=learning_rate, momentum=momentum)
            optimizer_sigmoid = torch.optim.SGD(two_layer_sigmoid_model.parameters(), lr=learning_rate,
                                                momentum=momentum)

            # Run experiment with ReLU activation
            print(
                f"Running experiment on ReLU activation with batch size: {batch_size} and learning rate: {learning_rate}")
            for t in range(epochs):
                print(f"Epoch {t + 1}\n-------------------------------")
                train_loop(train_loader, two_layer_relu_model, loss_fn_relu, optimizer_relu)
                test_loop(test_loader, two_layer_relu_model, loss_fn_relu)
                print("Done!")

            # Run experiment with Sigmoid activation
            print(
                f"Running experiment on Sigmoid activation with batch size: {batch_size} and learning rate: {learning_rate}")
            for t in range(epochs):
                print(f"Epoch {t + 1}\n-------------------------------")
                train_loop(train_loader, two_layer_sigmoid_model, loss_fn_sigmoid, optimizer_sigmoid)
                test_loop(test_loader, two_layer_sigmoid_model, loss_fn_sigmoid)
                print("Done!")


def question_four():
    batch_size = 1
    learning_rate = 0.01
    momentum = 0
    epochs = 2
    pollutation_rate = 0.1

    train_loader, test_loader = get_polluted_data(batch_size, pollutation_rate)
    loss_fn = nn.CrossEntropyLoss()
    two_layer_relu_model = TwoLayerNN()
    optimizer = torch.optim.SGD(two_layer_relu_model.parameters(), lr=learning_rate, momentum=momentum)

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_loader, two_layer_relu_model, loss_fn, optimizer)
        test_loop(test_loader, two_layer_relu_model, loss_fn)
        print("Done!")


def question_five():
    batch_size = 1
    learning_rate = 0.01
    momentum = 0
    epochs = 2

    # shifts tensor circular right
    train_loader, test_loader_right_shift = get_rotated_data(batch_size, -2, 2)

    # shifts tensor down
    _, test_loader_down_shift = get_rotated_data(batch_size, 2, 1)

    loss_fn = nn.CrossEntropyLoss()
    two_layer_relu_model = TwoLayerNN()
    optimizer = torch.optim.SGD(two_layer_relu_model.parameters(), lr=learning_rate, momentum=momentum)

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_loader, two_layer_relu_model, loss_fn, optimizer)
        print("Done!")

    print("Accuracy with circular right shift")
    test_loop(test_loader_right_shift, two_layer_relu_model, loss_fn)

    print("Accuracy with circular down shift")
    test_loop(test_loader_down_shift, two_layer_relu_model, loss_fn)


def main():
    question_two()
    #question_three()
    # question_four()
    # question_five()


if __name__ == '__main__':
    main()
