# create dataset
from torch.utils.data import Dataset
import pickle
import numpy as np
from matplotlib import pyplot as plt


class MyDataset(Dataset):
    """ex1 dataset."""

    def __init__(self, the_list, transform=None):
        self.the_list = the_list
        self.transform = transform

    def __len__(self):
        return len(self.the_list)

    def __getitem__(self, idx):
        item = self.the_list[idx]
        if self.transform:

            item = self.transform(item)
        return item

    def get_weights_to_sample(self):
        num_classes = len(label_names())
        amounts = [0] * num_classes
        for _, lable in self.the_list:
            amounts[lable] += 1
        tot_samples = sum(amounts)
        ratios = [tot_samples / amounts[i] for i in range(num_classes)]
        weights = [ratios[j] for _, j in self.the_list]
        return weights


def get_dataset_as_array(path='./data/dataset.pickle'):
    with open(path, 'rb') as f:
        dataset = pickle.load(f)
    return dataset


# useful for using data-loaders
def get_dataset_as_torch_dataset(path='./data/dataset.pickle', transform=None):
    dataset_as_array = get_dataset_as_array(path)
    dataset = MyDataset(dataset_as_array, transform)
    return dataset


# for visualizations
def un_normalize_image(img):
    img = img / 2 + 0.5
    img = np.array(img)
    img = np.transpose(img, (1, 2, 0))
    return img


def label_names():
    return {0: 'car', 1: 'truck', 2: 'cat'}


def save_data_as_img():
    data = get_dataset_as_torch_dataset('./data/train.pickle')
    for i in range(data.__len__()):
        cur_img, label = data.__getitem__(i)
        cur_img = un_normalize_image(cur_img)
        name = './data/old_data/' + str(label) + '/'
        name += "img" + str(i) + ".png"
        plt.imsave(name, cur_img)


def get_weights_to_fix(data):
    num_classes = len(label_names())
    amounts = [0] * num_classes
    for _, lable in data:
        amounts[lable] += 1
    tot_samples = sum(amounts)
    ratios = [tot_samples / amounts[i] for i in range(num_classes)]
    weights = [ratios[j] for _, j in data]
    return weights

# save_data_as_img()

