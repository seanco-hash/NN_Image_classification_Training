import torch
from torch.utils.data import DataLoader
from models import SimpleModel
import dataset
import torchvision
from training import get_transform


def evaluate_model_accuracy(net, testloader):
    """
    Evaluate by accuracy metric: total correct / total
    :param net:
    :param testloader:
    :return:
    """
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            sm = torch.nn.Softmax(dim=1)
            probabilities = sm(outputs)
            _, predicted = torch.max(probabilities.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the %d test images: %f %%' % (total,
            100 * correct / total))


def evaluate_accuracy_average(net, testloader):
    """
    Evaluation by average accuracy of the different classes.
    :param net:
    :param testloader:
    :return:
    """
    classes = ('car', 'truck', 'cat')
    class_correct = list(0. for i in range(3))
    class_total = list(0. for i in range(3))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            sm = torch.nn.Softmax(dim=1)
            probabilities = sm(outputs)
            _, predicted = torch.max(probabilities.data, 1)
            c = (predicted == labels)
            for i in range(c.shape[0]):
                label = labels[i]
                class_correct[label.item()] += c[i].item()
                class_total[label] += 1

    avg = 0
    for i in range(3):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))
        avg += 100 * class_correct[i] / class_total[i]

    avg /= 3
    total = sum(class_total)
    print('Average Accuracy of the network on the %d test images: %f %%' % (total, avg))


def evaluate_g_mean(net, testloader):
    """
    Evaluation by G-mean metric:
    (recall (cats) * recall (trucks) * recall (cars) ) ** 1/3
    :param net:
    :param testloader:
    :return:
    """
    class_correct = list(0. for i in range(3))
    false_negative = list(0. for i in range(3))
    class_total = list(0. for i in range(3))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            sm = torch.nn.Softmax(dim=1)
            probabilities = sm(outputs)
            _, predicted = torch.max(probabilities.data, 1)
            c = (predicted == labels)
            for i in range(c.shape[0]):
                label = labels[i]
                res = c[i].item()
                class_correct[label.item()] += res
                false_negative[label.item()] += (1-res)
                class_total[label] += 1

        recall_multiply = 1
        for i in range(3):
            recall_multiply *= (class_correct[i] / (class_correct[i] + false_negative[i]))
        recall_multiply = recall_multiply ** (1/3)
        print('G-Mean of the network on the %d test images: %f %%' % (sum(class_total), recall_multiply))


def evaluation(test_path= './data/dev.pickle', weights_path= './data/pre_trained.ckpt'):
    """
    Run evaluation process with the given original test data
    :param test_path:
    :param weights_path:
    :return:
    """
    data = dataset.get_dataset_as_torch_dataset(test_path)
    net = SimpleModel()
    net.load(weights_path)
    net.eval()
    testloader = DataLoader(data, batch_size=4, shuffle=False, num_workers=0)
    evaluate_model_accuracy(net, testloader)
    evaluate_accuracy_average(net, testloader)
    evaluate_g_mean(net, testloader)


def evaluate_fixed_data(test_path='./data/test_data', weights_path= './data/pre_trained.ckpt'):
    """
    Runs evaluation process with data saved as images and not in pickle.
    :param test_path:
    :param weights_path:
    :return:
    """
    trans = get_transform(False)
    net = SimpleModel()
    net.load(weights_path)
    net.eval()
    # testset = torchvision.datasets.ImageFolder(root=test_path, transform=torchvision.transforms.ToTensor())
    testset = torchvision.datasets.ImageFolder(root=test_path, transform=trans)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=0)
    evaluate_model_accuracy(net, testloader)
    evaluate_accuracy_average(net, testloader)
    evaluate_g_mean(net, testloader)


# evaluation(weights_path= './data/my_trained_model14.ckpt')
evaluate_fixed_data(weights_path= './data/my_trained_model15.ckpt')


