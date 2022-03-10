import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from torchvision import datasets, models, transforms
from copy import deepcopy
from torch.utils.data import Dataset
import kmeans

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


def get_custom_data(batch_size, size=1000):
    img_index = 0
    label_index = 1

    counter = np.zeros(10)

    custom_dataset = CustomDataset([], [])

    trainset, testset = get_data()

    for data in trainset:
        image = data[img_index]
        label = data[label_index]

        if counter[label] <= size:
            custom_dataset.data.append(deepcopy(image))
            custom_dataset.targets.append(deepcopy(label))
            counter[label] += 1

        if sum(counter) >= size*10:
            break

    trainloader = torch.utils.data.DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

    return trainloader, testloader


def get_data():
    trainset = datasets.MNIST('./data', download=True, train=True, transform=transforms.ToTensor())
    testset = datasets.MNIST('./data', download=True, train=False, transform=transforms.ToTensor())

    return trainset, testset


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

def get_feature_vectors(model, testloader):
    vectors = []
    for data in testloader:
        img, _ = data
        feature_vector = (model.encoder(img).reshape(len(img), 64)).detach().numpy()
        vectors.append(feature_vector)
    return vectors

def main():
    trainloader, testloader = get_custom_data(64)
    train_data, train_labels, test_data, test_labels = kmeans.get_custom_data()
    train_data = train_data.reshape(len(train_data), -1)
    test_data = test_data.reshape(len(test_data), -1)

    model = Autoencoder()
    max_epochs = 2
    outputs = train(model, trainloader, num_epochs=max_epochs)

    kmeans_model = kmeans.KMeans(init="k-means++", n_clusters=10, n_init=4, random_state=0)

    vectors = get_feature_vectors(model, testloader)
    #
    kmeans_model = kmeans_model.fit(train_data)
    results = kmeans_model.predict(vectors)
    acc = kmeans.accuracy(test_labels, results)
    print(acc)

    # for k in range(0, max_epochs, 5):
    #     plt.figure(figsize=(9, 2))
    #     for i, item in enumerate(vectors):
    #         if i >= 9: break
    #         plt.subplot(2, 9, 9 + i + 1)
    #         plt.imshow(item[0])
    # plt.show()

    for k in range(0, max_epochs, 5):
        plt.figure(figsize=(9, 2))
        imgs = outputs[k][1].detach().numpy()
        recon = outputs[k][2].detach().numpy()
        for i, item in enumerate(imgs):
            if i >= 9: break
            plt.subplot(2, 9, i + 1)
            plt.imshow(item[0])

        for i, item in enumerate(recon):
            if i >= 9: break
            plt.subplot(2, 9, 9 + i + 1)
            plt.imshow(item[0])

    plt.show()


if __name__ == '__main__':
    main()
