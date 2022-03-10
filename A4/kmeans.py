import numpy as np

from torch.utils.data import Dataset
from torchvision import datasets, models, transforms

from time import time
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import seaborn as sns;

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


def get_custom_data(size=100):
    img_index = 0
    label_index = 1

    counter = np.zeros(10)

    custom_train_data = []
    custom_train_targets = []

    custom_test_data = []
    custom_test_targets = []

    trainset, testset = get_data()

    for i in range(0, len(trainset.data)):
        label = trainset.targets[i].numpy()
        data = ((trainset.data[i].numpy()).astype('float32'))

        if counter[label] <= size:
            custom_train_data.append(data)
            custom_train_targets.append(label)
            counter[label] += 1

        if sum(counter) >= size*10:
            break

    for i in range(0, len(testset.data)):
        label = testset.targets[i].numpy()
        data = ((testset.data[i].numpy()).astype('float32'))

        custom_test_data.append(data)
        custom_test_targets.append(label)


    return np.asarray(custom_train_data), np.asarray(custom_train_targets), np.asarray(custom_test_data), np.asarray(custom_test_targets)


def get_data():

    trainset = datasets.MNIST('./data', download=True, train=True) #transform=data_transforms['train'])
    testset = datasets.MNIST('./data', download=True, train=False) #transform=data_transforms['val'])

    return trainset, testset


def accuracy(true_row_labels, predicted_row_labels):
    """Get the best accuracy.

    Parameters
    ----------
    true_row_labels: array-like
        The true row labels, given as external information
    predicted_row_labels: array-like
        The row labels predicted by the model

    Returns
    -------
    float
        Best value of accuracy
    """

    cm = confusion_matrix(true_row_labels, predicted_row_labels)
    sns.set()

    indexes = linear_sum_assignment(_make_cost_m(cm))
    (row, column) = indexes

    sorted_cm = []
    for _, r in sorted(zip(column, cm), key=lambda x: x[0]):
        sorted_cm.append(r)

    ax = sns.heatmap(sorted_cm, annot=True, fmt="d", cmap="Blues")
    plt.show()

    total = 0
    for i in range(0, len(row)):
        r = row[i]
        c = column[i]
        value = cm[r][c]
        total += value

    return (total * 1. / np.sum(cm))

def _make_cost_m(cm):
    s = np.max(cm)
    return (- cm + s)


def bench_k_means(kmeans, name, data, labels, test_data, test_labels):
    """Benchmark to evaluate the KMeans initialization methods.

    Parameters
    ----------
    kmeans : KMeans instance
        A :class:`~sklearn.cluster.KMeans` instance with the initialization
        already set.
    name : str
        Name given to the strategy. It will be used to show the results in a
        table.
    data : ndarray of shape (n_samples, n_features)
        The data to cluster.
    labels : ndarray of shape (n_samples,)
        The labels used to compute the clustering metrics which requires some
        supervision.
    """
    t0 = time()
    estimator = make_pipeline(StandardScaler(), kmeans).fit(data)
    fit_time = time() - t0
    results = [name, fit_time, estimator[-1].inertia_]

    # Define the metrics which require only the true labels and estimator
    # labels
    clustering_metrics = [
        metrics.homogeneity_score,
        metrics.completeness_score,
        metrics.v_measure_score,
        metrics.adjusted_rand_score,
        metrics.adjusted_mutual_info_score,
    ]
    results += [m(labels, estimator[-1].labels_) for m in clustering_metrics]

    # The silhouette score requires the full dataset
    results += [
        metrics.silhouette_score(
            data,
            estimator[-1].labels_,
            metric="euclidean",
            sample_size=300,
        )
    ]

    # Show the results
    formatter_result = (
        "{:9s}\t{:.3f}s\t{:.0f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}"
    )
    print(formatter_result.format(*results))


def retrieve_info(cluster_labels, y_train, kmeans):
    # Initializing
    reference_labels = {}
    # For loop to run through each label of cluster label
    for i in range(len(np.unique(kmeans.labels_))):
        index = np.where(cluster_labels == i, 1, 0)
    num = np.bincount(y_train[index == 1]).argmax()
    reference_labels[i] = num

    return reference_labels

def calculate_metrics(model,output):
    print("Number of clusters is {}".format(model.n_clusters))
    print("Inertia : {}".format(model.inertia_))
    print("Homogeneity :       {}".format(metrics.homogeneity_score(output,model.labels_)))


def main():
    train_data, train_labels, test_data, test_labels = get_custom_data()
    train_data = train_data.reshape(len(train_data), -1)
    test_data = test_data.reshape(len(test_data), -1)

    (n_samples, n_features), n_digits = train_data.shape, np.unique(train_labels).size

    print(f"# digits: {n_digits}; # samples: {n_samples}; # features {n_features}")
    print(82 * "_")
    print("init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette")

    kmeans = KMeans(init="k-means++", n_clusters=n_digits, n_init=4, random_state=0)
    bench_k_means(kmeans=kmeans, name="k-means++", data=train_data, labels=train_labels,
                  test_data=test_data, test_labels=test_labels)

    kmeans = kmeans.fit(train_data)
    results = kmeans.predict(test_data)
    acc = accuracy(test_labels, results)
    print(acc)


if __name__ == '__main__':
    main()
