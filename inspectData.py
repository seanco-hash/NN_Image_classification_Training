from torch.utils.data import Dataset
import pickle
import numpy as np
import matplotlib.pyplot as plt
import dataset


def inspect():
    """
    Shows random images
    """
    train_path = './data/train.pickle'
    datush = dataset.get_dataset_as_torch_dataset(train_path)
    item = datush.__getitem__(244)
    item = item[0]
    item = dataset.un_normalize_image(item)
    plot2x2Array(item)


def countLabels(data):
    """
    Counts amount of samples from each class
    :param data:
    :return:
    """
    n = data.__len__()
    print(n)
    sizes = [0, 0, 0]
    for i in range(n):
        item = data.__getitem__(i)
        idx = item[1]
        sizes[idx] += 1
    for j in range(3):
        print(sizes [j])
    labels = 'Cars', 'Tracks', 'Cats'
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')
    plt.title("Amount of Items From Each Label")
    plt.show()


def checkLabelsAppearance(data):
    """
    Check location of appearance of samples from different classes
    along the dataset
    :param data:
    :return:
    """
    n = data.__len__()
    names = dataset.label_names()
    types = [[] for i in range(3)]
    for i in range(n):
        idx = data.__getitem__(i)[1]
        types[idx].append(i)
    fig, ax = plt.subplots()
    i = 0
    for color in ['blue', 'orange', 'green']:
        x = types[i]
        y = [i] * len(x)
        scale = 7
        ax.scatter(x, y, c=color, s=scale, label=names[i],
                   alpha=0.3, edgecolors='none')
        i += 1

    ax.legend(loc='upper right', title="Labels")
    ax.grid(True)
    plt.title("Labels Distribution")
    plt.show()


def plot2x2Array(image):
    imgplot = plt.imshow(image)
    plt.show()


def checkMeanAndStd(data):
    """
    Searching for outliers by calculating mean and std
    :param data:
    :return:
    """
    lst = []
    bads = 0
    for i in range(data.__len__()):
        item = data.__getitem__(i)[0]
        item = dataset.un_normalize_image(item)
        cur_mean = np.mean(item)
        cur_std = np.std(item)
        if cur_mean < 0.1 or cur_mean > 0.9 or cur_std < 0.1:
            bads += 1
            lst.append(i)
    print(bads)
    print(lst)

#
# train_path = './data/train.pickle'
# datush = dataset.get_dataset_as_torch_dataset(train_path)
# checkMeanAndStd(datush)
# checkLabelsAppearance(datush)
# inspect()
# countLabels(datush)