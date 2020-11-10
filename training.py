import torch
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from models import SimpleModel
import dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from sampler import MultilabelBalancedRandomSampler
import numpy as np
from torchvision import transforms
import torch.optim.lr_scheduler


def get_transform(is_train):
    """
    I tried to use transformations, I didn't found it useful but here what I tried
    :param is_train:
    :return:
    """
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    if is_train:
        transforms_val = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            # transforms.Pad(25, padding_mode='symmetric'),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(10),
            transforms.ToTensor(),
            # transforms.Normalize(mean, std),
            # transforms.RandomErasing(p=0.75, scale=(0.02, 0.1), value=1.0, inplace=False)
    ])
    else:
        transforms_val = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean, std)
        ])
    return transforms_val


def get_one_hot(targets, nb_classes):
    """
    Converts labels to one hot matrix - needed fot using the external sampler package
    :param targets:
    :param nb_classes:
    :return:
    """
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])


def do_train(optimizer, criterion, epochs, trainloader, net, scheduler):
    """
    Trains given model with given data
    :return:
    """
    ep = 0
    count = [0] * 3
    for epoch in range(epochs):
        print("epoch: %d" % ep)
        ep += 1
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            for label in labels:
                count[label] += 1
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 1000 == 999:  # print every 1000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 1000))
                running_loss = 0.0
        scheduler.step()

    print(count)
    print('Finished Training')


def train_model(train_path= './data/train.pickle', is_fixed=False, lr=0.02, is_decay_lr=True):
    """
    Loads model, set the data and parameters and run training.
    I left uncomment lines to show the process I made.
    @:param is_fixed: the data format. If false - the original pickle file.
    """
    trans = get_transform(True)
    if is_fixed:
        data = torchvision.datasets.ImageFolder(root=train_path, transform=trans)
        # lables = get_one_hot(np.array(data.targets), 3)
        # train_sampler = MultilabelBalancedRandomSampler(lables, class_choice="least_sampled")
        weights = dataset.get_weights_to_fix(data)
        train_sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights))
        train_loader = DataLoader(data, batch_size=4, shuffle=False, num_workers=0, sampler=train_sampler)
    else:
        data = dataset.get_dataset_as_torch_dataset(train_path)
        # data.transform = trans
        # labels = []
        # for i in range(data.__len__()):
        #     _, label = data.__getitem__(i)
        #     labels.append(label)
        # labels = get_one_hot(np.array(labels), 3)
        # train_sampler = MultilabelBalancedRandomSampler(labels, class_choice="least_sampled")
        weights = data.get_weights_to_sample()
        train_sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights))
        train_loader = DataLoader(data, batch_size=4, shuffle=True, num_workers=0, sampler=train_sampler)
    net = SimpleModel()
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.8, weight_decay=0.01)
    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.01)
    net.train()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    do_train(optimizer, criterion, 100, train_loader, net, scheduler)
    net.save("./data/my_trained_model15.ckpt")


# train_model()
# torch.multiprocessing.freeze_support()
train_model(train_path="./data/data_new", is_fixed=True)
# train_model(train_path="./data/old_data", is_fixed=True)
