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

    num_polluted_images = int(num_pixels * pollution_rate/2)

    for i in range(num_polluted_images):
        y_coord = random.randint(0, row - 1)
        x_coord = random.randint(0, col - 1)
        img[y_coord][x_coord] = 255

        y_coord = random.randint(0, row - 1)
        x_coord = random.randint(0, col - 1)
        img[y_coord][x_coord] = 0

    return img

def get_custom_data(dataset, batch_size=64):
    img_index = 0
    label_index = 1

    custom_train_dataset = CustomDataset()

    convert_tensor = transforms.ToTensor()
    pollution_rate = .1
    print("Creating polluted data with rate: {}", pollution_rate)
    for data in dataset:
        label = data[label_index]
        img = np.asarray(data[img_index])

        polluted_img = add_noise(img, pollution_rate)
        tensor_img = convert_tensor(polluted_img)

        custom_train_dataset.data.append(tensor_img)
        custom_train_dataset.targets.append(label)

    dataloader = torch.utils.data.DataLoader(custom_train_dataset, batch_size=batch_size, shuffle=True)

    return dataloader

def get_data():
    trainset = datasets.MNIST('./data', download=True, train=True)
    testset = datasets.MNIST('./data', download=True, train=False)

    return trainset, testset

def test(model, testset):
    output = []

    testloader = get_custom_data(testset)

    for i, (data, img) in enumerate(zip(testloader, testset)):
        if i >= 10: break
        print(i)
        polluated_img, _  = data
        recon = model(polluated_img)
        output.append((img, polluated_img, recon))

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

def main():
    trainset, testset = get_data()
    trainloader = get_custom_data(trainset)
    model = Autoencoder()
    max_epochs = 2
    outputs = train(model, trainloader, num_epochs=max_epochs)
    imgs = test(model, testset)

    return imgs

    # for k in range(0, max_epochs, 5):
    #     plt.figure(figsize=(9, 2))
    #     imgs = outputs[k][1].detach().numpy()
    #     recon = outputs[k][2].detach().numpy()
    #     for i, item in enumerate(imgs):
    #         if i >= 9: break
    #         plt.subplot(2, 9, i + 1)
    #         plt.imshow(item[0])
    #
    #     for i, item in enumerate(recon):
    #         if i >= 9: break
    #         plt.subplot(2, 9, 9 + i + 1)
    #         plt.imshow(item[0])
    #
    # plt.show()

if __name__ == '__main__':
    main()
