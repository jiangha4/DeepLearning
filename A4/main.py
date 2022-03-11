import copy

import numpy as np
import random
import cv2
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from torchvision import datasets, models, transforms


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential( # like the Composition layer you built
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 7)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class CustomDataset(Dataset):
    def __init__(self, data=[], targets=[]):
        self.data = data
        self.targets = targets

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        return x, y

    def __len__(self):
        return len(self.data)


def add_noise(img, pollution_rate):
    row, col = img.shape
    num_pixels = row * col

    num_polluted_images = int((num_pixels * pollution_rate)/2)

    for i in range(num_polluted_images):
        y_coord = random.randint(0, row - 1)
        x_coord = random.randint(0, col - 1)
        img[y_coord][x_coord] = 255

        y_coord = random.randint(0, row - 1)
        x_coord = random.randint(0, col - 1)
        img[y_coord][x_coord] = 0

    return img


def get_custom_data(dataset, pollution_rate, batch_size=64):
    img_index = 0
    label_index = 1

    custom_train_dataset = CustomDataset()
    custom_dataset = CustomDataset()

    convert_tensor = transforms.ToTensor()
    print("Creating polluted data with rate: {}".format(pollution_rate))
    for data in dataset:
        label = copy.deepcopy(data[label_index])
        pimg = copy.deepcopy(np.asarray(data[img_index]))
        img = copy.deepcopy(np.asarray(data[img_index]))

        polluted_img = add_noise(pimg, pollution_rate)
        polluted_tensor_img = convert_tensor(polluted_img)
        tensor_img = convert_tensor(img)

        custom_train_dataset.data.append(polluted_tensor_img)
        custom_train_dataset.targets.append(label)

        custom_dataset.data.append(tensor_img)
        custom_dataset.targets.append(label)

    polluted_dataloader = torch.utils.data.DataLoader(custom_train_dataset, batch_size=batch_size)
    normal_dataloader = torch.utils.data.DataLoader(custom_dataset, batch_size=batch_size)

    return polluted_dataloader, normal_dataloader


def get_data():
    trainset = datasets.MNIST('./data', download=True, train=True)
    testset = datasets.MNIST('./data', download=True, train=False)

    return trainset, testset


def test(model, testset, polluation_rate):
    output = []
    batch_size = 64
    polluted_dataloader, normal_dataloader = get_custom_data(testset, polluation_rate, batch_size=batch_size)

    for i, (pdata, data) in enumerate(zip(polluted_dataloader, normal_dataloader)):
        if i >= 2: break
        img, _ = data
        polluted_img, _ = pdata
        recon = model(polluted_img)
        output.append((img, polluted_img, recon),)

    return output


def train(model, trainloader, num_epochs=5, learning_rate=1e-3):
    torch.manual_seed(42)
    criterion = nn.MSELoss() # mean square error loss
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=1e-5)
    outputs = []
    for epoch in range(num_epochs):
        for data in trainloader:
            img, _ = data
            recon = model(img)
            loss = criterion(recon, img)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print('Epoch:{}, Loss:{:.4f}'.format(epoch+1, float(loss)))
        outputs.append((epoch, img, recon),)
    return outputs


def create_plots(image_vector):
    for img, pollutated_img, recon in image_vector:
        plt.figure(figsize=(10, 3))
        nimg = img.detach().numpy()
        pimg = pollutated_img.detach().numpy()
        recon = recon.detach().numpy()

        j = 0
        for i, item in enumerate(nimg):
            if i >= 10: break
            if i % 2 != 1: continue
            else:
                plt.subplot(3, 5, j + 1)
                plt.imshow(item[0])
                j += 1

        j = 0
        for i, item in enumerate(pimg):
            if i >= 10: break
            if i % 2 != 0: continue
            else:
                plt.subplot(3, 5, 5 + j + 1)
                plt.imshow(item[0])
                j += 1

        j = 0
        for i, item in enumerate(recon):
            if i >= 10: break
            if i % 2 != 0: continue
            else:
                plt.subplot(3, 5, 10 + j + 1)
                plt.imshow(item[0])
                j += 1

    plt.show()


def train_model(model, trainset, pollution_rate):
    polluated_trainloader, normal_trainloader = get_custom_data(trainset, pollution_rate=pollution_rate)
    max_epochs = 20
    train(model, polluated_trainloader, num_epochs=max_epochs)

def main():
    pollution_rate = [.2, .4, .6, .8, 1]
    model = Autoencoder()
    trainset, testset = get_data()
    for rate in pollution_rate:
        train_model(model, trainset, rate)
        imgs = test(model, testset, rate)
        create_plots(imgs)

if __name__ == '__main__':
    main()
