import torch
from torch.utils.data import DataLoader
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
from training import get_transform
import matplotlib.pyplot as plt


def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])


def do_train(optimizer, criterion, epochs, trainloader, net):
    ep = 0
    loss_list = []
    for epoch in range(epochs):
        print("epoch: %d" % ep)
        ep += 1
        running_loss = 0.0
        running_loss2 = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_loss2 += loss.item()
            if i % 1000 == 999:  # print every 1000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 1000))
                running_loss = 0.0
                loss_list.append(loss.item())

    print('Finished Training')
    return loss_list


def analyze(train_path= './data/train.pickle', lr=0.02):
    # trans = get_transform(True)
    data = torchvision.datasets.ImageFolder(root=train_path, transform=torchvision.transforms.ToTensor())
    lables = get_one_hot(np.array(data.targets), 3)
    train_sampler = MultilabelBalancedRandomSampler(lables, class_choice="least_sampled")
    train_loader = DataLoader(data, batch_size=4, shuffle=False, num_workers=0, sampler=train_sampler)
    net = SimpleModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr)
    net.train()
    loss_list = do_train(optimizer, criterion, 120, train_loader,  net)
    return loss_list


def analyze_lr():
    """
    Runs analysis for some epsilons and plots results
    :return:
    """
    lr = [0.5, 0.05, 0.005, 0.0005, 0.00005]
    all_loss_lists = []
    for i in range(5):
        # all_loss_lists.append([j/(100*(i+1)) for j in range(100)])
        all_loss_lists.append(analyze(train_path="./data/old_data", lr=lr[i]))

    n = len(all_loss_lists[0])
    names = lr
    fig, ax = plt.subplots()
    i = 0
    for color in ['blue', 'orange', 'green', 'yellow', 'red']:
        x = range(n)
        # y = np.log(all_loss_lists[i])
        y = all_loss_lists[i]
        for k in range(n):
            if y[k] > 3:
                y[k] = 3
        ax.plot(x, y, c=color, label=names[i], alpha=0.4)
        # ax.axis([0, n, 0, 3])
        i += 1

    ax.legend(loc='upper right', title="Learning Rates")
    ax.grid(True)
    plt.title("Loss / Time by Learning Rates")
    plt.show()

    for k in range(5):
        print(len(all_loss_lists[k]))


analyze_lr()
