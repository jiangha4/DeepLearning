import random
import torch
from copy import deepcopy
from torch.utils.data import Dataset


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


def create_polluted_data(dataset, pollutionRate):
    img_index = 0
    label_index = 1

    pollutedDataset = CustomDataset([], [])
    classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

    # Pollute the data set
    numPollutedImages = len(dataset) * pollutionRate

    #Generate a list of 1% of the dataset of randomly sampled ints from 0 to len - 1 of the data training set
    indexes = random.sample(range(len(dataset)-1), int(numPollutedImages))

    print("Generating polluted data...")
    for index in range(0, len(dataset)):
        pollutedDataset.data.append(deepcopy(dataset[index][img_index]))
        newIndexValue = deepcopy(dataset[index][label_index])
        if index in indexes:
            # pollute the dataset by shifting the class value up by 1
            newIndexValue = newIndexValue + 1
            # if index value looped past the len of the classes
            # set it to 0
            if newIndexValue > len(classes) - 1:
                newIndexValue = 0

        pollutedDataset.targets.append(newIndexValue)
    print("Finished generating polluted data!")
    return pollutedDataset


def rotate_dataset(dataset, shift, dim):
    # dim 1 shifts top and down
    # dim 2 shift left and right

    img_index = 0
    label_index = 1
    rotated_dataset = CustomDataset([], [])
    print("Rotating pixels...")
    for data in dataset:
        rotated_tensor = torch.roll(data[img_index], shift, dim)
        rotated_dataset.data.append(rotated_tensor)
        rotated_dataset.targets.append(data[label_index])

    print("Rotation complete!")
    return rotated_dataset
