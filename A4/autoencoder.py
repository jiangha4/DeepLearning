import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from torchvision import datasets, models, transforms
from copy import deepcopy
from torch.utils.data import Dataset
import kmeans
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


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

        if sum(counter) >= size * 10:
            break

    trainloader = torch.utils.data.DataLoader(custom_dataset, batch_size=batch_size)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size)

    return trainloader, testloader


def get_data():
    trainset = datasets.MNIST('./data', download=True, train=True, transform=transforms.ToTensor())
    testset = datasets.MNIST('./data', download=True, train=False, transform=transforms.ToTensor())

    return trainset, testset


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(  # like the Composition layer you built
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
    criterion = nn.MSELoss()  # mean square error loss
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

        print('Epoch:{}, Loss:{:.4f}'.format(epoch + 1, float(loss)))
        outputs.append((epoch, img, recon), )
    return outputs


def get_feature_vectors(model, testloader):
    for i, data in enumerate(testloader):
        img, _ = data
        feature_vector = (model.encoder(img).reshape(len(img), 64)).detach().numpy()
        if i == 0:
            vectors = feature_vector
        else:
            vectors = np.concatenate((vectors, feature_vector), axis=0)
    return vectors

def run_feature_map_kmeans(train_vectors, test_vectors, test_labels):

    kmeans_model = kmeans.KMeans(init="k-means++", n_clusters=10, n_init=4, random_state=0)

    kmeans_model = kmeans_model.fit(train_vectors)
    results = kmeans_model.predict(test_vectors)
    acc = kmeans.accuracy(test_labels, results)
    print(acc)

def run_pca(train_vectors, test_vectors, test_labels):
    pca = PCA(n_components=10).fit(train_vectors)
    pca_test_vectors = pca.fit_transform(test_vectors)
    pca_train_vectors = pca.fit_transform(train_vectors)

    kmeans_model = KMeans(init="k-means++", n_clusters=10, n_init=4)

    kmeans_model = kmeans_model.fit(pca_train_vectors)
    results = kmeans_model.predict(pca_test_vectors)
    acc = kmeans.accuracy(test_labels, results)
    print(acc)

def main():
    trainloader, testloader = get_custom_data(64)
    _, _, _, test_labels = kmeans.get_custom_data()

    model = Autoencoder()
    max_epochs = 20
    train(model, trainloader, num_epochs=max_epochs)

    train_vectors = get_feature_vectors(model, trainloader)
    test_vectors = get_feature_vectors(model, testloader)

    # part a
    run_feature_map_kmeans(train_vectors, test_vectors, test_labels)

    # part b
    run_pca(train_vectors, test_vectors, test_labels)


if __name__ == '__main__':
    main()
