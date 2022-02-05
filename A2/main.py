import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import models
import time

def get_data(batch_size):
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)

    testset = datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    dataset_sizes = {'train': len(trainset), 'test': len(testset)}

    return trainloader, testloader, dataset_sizes

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

def train_mse(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        corrected_preds = (torch.max(pred, dim=1)[1]).float()

        loss = loss_fn(corrected_preds, y.float())
        loss.requires_grad = True

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return loss.item()

def test_mse(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            corrected_preds = (torch.max(pred, dim=1)[1]).float()
            test_loss += loss_fn(corrected_preds, y.float()).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, correct

def plot_data(x_data, y_data, x_axis, y_axis, title, file):
    fig, ax = plt.subplots()
    ax.plot(x_data, y_data)

    ax.set(xlabel=x_axis, ylabel=y_axis, title=title)
    fig.savefig(file)

def cross_entropy_loss(model, learning_rate, epochs, trainloader, testloader):
    training_loss = list()
    testing_loss = list()
    accuracy_list = list()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(0, epochs):
        print("Epoch %s:" % epoch)
        train_loss = train(trainloader, model, loss_fn, optimizer)
        training_loss.append(train_loss)
        test_loss, accuracy = test(testloader, model, loss_fn)
        testing_loss.append((test_loss))
        accuracy_list.append(accuracy)

    return training_loss, testing_loss, accuracy_list

def mean_square_loss(model, learning_rate, epochs, trainloader, testloader):
    training_loss = list()
    testing_loss = list()
    accuracy_list = list()
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(0, epochs):
        print("Epoch %s:" % epoch)
        train_loss = train_mse(trainloader, model, loss_fn, optimizer)
        training_loss.append(train_loss)
        test_loss, accuracy = test_mse(testloader, model, loss_fn)
        testing_loss.append((test_loss))
        accuracy_list.append(accuracy)

    return training_loss, testing_loss, accuracy_list


def sigmoid_activation(learning_rate, epochs, train_loader, test_loader, x_axis):
    model_cel = models.LeNetSigmoid()
    file_template = "graphs6/s-%s-lr-%s-%s.png"
    loss_fn = "cel"
    # Cross Entropy Loss
    training_loss, testing_loss, accuracy_list = cross_entropy_loss(model_cel, learning_rate, epochs, train_loader,
                                                                    test_loader)

    file_name = file_template % (loss_fn, str(learning_rate), "training")
    plot_data(x_axis, training_loss, "Epochs", "Training Loss", "Sigmoid Activation Training Loss, Learning Rate: %s" % str(learning_rate), file_name)

    file_name = file_template % (loss_fn, str(learning_rate), "testing")
    plot_data(x_axis, testing_loss, "Epochs", "Testing Loss", "Sigmoid Activation Testing Loss, Learning Rate: %s" % str(learning_rate), file_name)

    file_name = file_template % (loss_fn, str(learning_rate), "accuracy")
    plot_data(x_axis, accuracy_list, "Epochs", "Accuracy", "Sigmoid Activation Accuracy, Learning Rate: %s" % str(learning_rate), file_name)

    model_mse = models.LeNetSigmoid()
    training_loss, testing_loss, accuracy_list = mean_square_loss(model_mse, learning_rate, epochs, train_loader, str(learning_rate))
    loss_fn = "mse"

    # Mean Square Error Loss
    file_name = file_template % (loss_fn, str(learning_rate), "training")
    plot_data(x_axis, training_loss, "Epochs", "Training Loss", "Sigmoid Activation Training Loss, Learning Rate: %s" % str(learning_rate), file_name)

    file_name = file_template % (loss_fn, str(learning_rate), "testing")
    plot_data(x_axis, testing_loss, "Epochs", "Testing Loss", "Sigmoid Activation Testing Loss, Learning Rate: %s" % str(learning_rate), file_name)

    file_name = file_template % (loss_fn, str(learning_rate), "accuracy")
    plot_data(x_axis, accuracy_list, "Epochs", "Accuracy", "Sigmoid Activation Accuracy, Learning Rate: %s" % str(learning_rate), file_name)


def tanh_activation(learning_rate, epochs, train_loader, test_loader, x_axis):
    model_cel = models.LeNetTanh()
    file_template = "graphs6/t-%s-lr-%s-%s.png"
    loss_fn = "cel"
    # Cross Entropy Loss
    training_loss, testing_loss, accuracy_list = cross_entropy_loss(model_cel, learning_rate, epochs, train_loader,
                                                                    test_loader)

    if learning_rate == 0.001:
        torch.save(model_cel, "saved_model")

    file_name = file_template % (loss_fn, str(learning_rate), "training")
    plot_data(x_axis, training_loss, "Epochs", "Training Loss", "Tanh Activation Training Loss, Learning Rate: %s" % str(learning_rate), file_name)

    file_name = file_template % (loss_fn, str(learning_rate), "testing")
    plot_data(x_axis, testing_loss, "Epochs", "Testing Loss", "Tanh Activation Testing Loss, Learning Rate: %s" % str(learning_rate), file_name)

    file_name = file_template % (loss_fn, str(learning_rate), "accuracy")
    plot_data(x_axis, accuracy_list, "Epochs", "Testing Loss", "Tanh Activation Accuracy, Learning Rate: %s" % str(learning_rate), file_name)

    model_mse = models.LeNetTanh()
    loss_fn = "mse"
    training_loss, testing_loss, accuracy_list = mean_square_loss(model_mse, learning_rate, epochs, train_loader, test_loader)

    file_name = file_template % (loss_fn, str(learning_rate), "training")
    plot_data(x_axis, training_loss, "Epochs", "Training Loss", "Tanh Activation Training Loss, Learning Rate: %s" % str(learning_rate), file_name)

    file_name = file_template % (loss_fn, str(learning_rate), "training")
    plot_data(x_axis, testing_loss, "Epochs", "Testing Loss", "Tanh Activation Testing Loss, Learning Rate: %s" % str(learning_rate), file_name)

    file_name = file_template % (loss_fn, str(learning_rate), "training")
    plot_data(x_axis, accuracy_list, "Epochs", "Accuracy", "Tanh Activation Accuracy, Learning Rate: %s" % str(learning_rate), file_name)



def question_one(rate, epochs, train_loader, test_loader, x_axis):
    sigmoid_activation(rate, epochs, train_loader, test_loader, x_axis)
    tanh_activation(rate, epochs, train_loader, test_loader, x_axis)


def question_two(epochs, train_loader, test_loader, x_axis):
    model = models.LeNetReLu()
    learning_rate = 0.001

    file_template = "graphs4/reLu-%s-lr-%s-%s.png"
    loss_fn = "cel"
    # Cross Entropy Loss
    file_name = file_template % (loss_fn, str(learning_rate), "training")
    training_loss, testing_loss, accuracy_list = cross_entropy_loss(model, learning_rate, epochs, train_loader,
                                                                     test_loader)
    plot_data(x_axis, training_loss, "Epochs", "Training Loss", "ReLu Activation Training Loss", file_name)

    file_name = file_template % (loss_fn, str(learning_rate), "testing")
    plot_data(x_axis, testing_loss, "Epochs", "Testing Loss", "ReLu Activation Testing Loss", file_name)

    file_name = file_template % (loss_fn, str(learning_rate), "accuracy")
    plot_data(x_axis, accuracy_list, "Epochs", "Accuracy", "ReLu Activation Accuracy", file_name)


def question_three(epochs, train_loader, test_loader, x_axis):
    model = models.FiveLayerCNN()
    learning_rate = 0.01

    file_template = "graphs4/5layerCNN-%s.png"

    start = time.process_time()
    training_loss, testing_loss, accuracy_list = cross_entropy_loss(model, learning_rate, epochs, train_loader,
                                                                     test_loader)
    end = time.process_time()

    print("Total time: %s", str(end - start))

    file_name = file_template % ("training")
    plot_data(x_axis, training_loss, "Epochs", "Training Loss", "5 Layer CNN Training Loss", file_name)

    file_name = file_template % ("testing")
    plot_data(x_axis, testing_loss, "Epochs", "Testing Loss", "5 Layer CNN Testing Loss", file_name)

    file_name = file_template % ("accuracy")
    plot_data(x_axis, accuracy_list, "Epochs", "Accuracy", "5 Layer CNN Accuracy", file_name)


def main():
    learning_rate = [0.1, 0.01, 0.001]
    # learning_rate = [0.1]
    epochs = 50
    x_axis = np.arange(1, epochs+1, step=1)
    train_loader, test_loader, dataset_sizes = get_data(30)

    for rate in learning_rate:
        question_one(rate, epochs, train_loader, test_loader, x_axis)

    #question_two(epochs, train_loader, test_loader, x_axis)
    #question_three(epochs, train_loader, test_loader, x_axis) - # 2045.68132 seconds


if __name__ == "__main__":
    main()
