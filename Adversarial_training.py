import torch
import torchvision
from torch.utils.data import DataLoader
from models import SimpleModel
import dataset
import torch.nn.functional as F
import torch.optim.lr_scheduler
from torchvision import transforms
import matplotlib.pyplot as plt
from training import get_transform


def apply_noise_single_im(im, eps, grads):
    """
    Image + ( sign(grads) * epsilon ))
    :param im: original image
    :param eps: epsilon - strength of noise
    :param grads: gradients from prediction
    :return:
    """
    sign = grads.sign()
    noised_image = im + (eps * sign)
    noised_image = torch.clamp(noised_image, 0, 1)
    return noised_image


def show_examples(examples, epsilons):
    """
    Plot chosen adversarial examples by epsilons we used
    :param examples:
    :param epsilons:
    :return:
    """
    cnt = 0
    for i in range(len(epsilons)):
        for j in range(len(examples[i])):
            cnt += 1
            plt.subplot(len(epsilons),len(examples[0]),cnt)
            plt.xticks([], [])
            plt.yticks([], [])
            if j == 0:
                plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
            orig, adv, ex = examples[i][j]
            ex = dataset.un_normalize_image(ex)
            plt.title("{} -> {}".format(orig, adv))
            plt.imshow(ex, cmap="gray")

    plt.tight_layout()
    plt.show()


def optimize_noise(testloader, net, init_noise, lr):
    """
    Implementation of noise optimization (the second method I mentioned in the report)
    """
    adversarial_succ = []
    classes = ('car', 'truck', 'cat')
    class_correct = list(0. for i in range(3))
    class_total = list(0. for i in range(3))
    correct_total = 0
    k = 0
    sm = torch.nn.Softmax(dim=1)

    # Noise optimization phase
    for data in testloader:
        with torch.no_grad():
            images, labels = data
            outputs1 = net(images)
            probabilities = sm(outputs1)
            _, predicted = torch.max(probabilities.data, 1)
            correct = predicted.item() == labels.item()
            if not correct:
                continue
        noised_im = init_noise + images
        noised_im = torch.clamp(noised_im, 0, 1)
        noised_im.requires_grad = True
        outputs = net(noised_im)
        wrong_label = labels
        wrong_label = (wrong_label + 1) % 3
        loss = F.nll_loss(outputs, wrong_label)
        loss.backward()
        grads = noised_im.grad.data

        init_noise -= (lr * grads[0])

    # Prediction of image + loss phase
    for data in testloader:
        images, labels = data
        noised_im = init_noise + images
        noised_im = torch.clamp(noised_im, 0, 1)
        new_output = net(noised_im)
        probabilities = sm(new_output)
        _, new_predicted = torch.max(probabilities.data, 1)
        correct = new_predicted.item() == labels.item()
        class_total[labels] += 1
        k += 1
        if not correct:
            if len(adversarial_succ) < 5 and k % 2 == 0:
                noised_ex = noised_im.squeeze().detach().cpu().numpy()
                adversarial_succ.append((0, new_predicted.item(), noised_ex))
        else:
            correct_total += 1
            class_correct[new_predicted] += 1

    avg = 0
    for i in range(3):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))
        avg += 100 * class_correct[i] / class_total[i]

    avg /= 3
    total = sum(class_total)
    accuracy = 100 * correct_total / total
    print('Average Accuracy of the network on the %d test images: %f %%' % (total, avg))
    print('Accuracy of the network on the %d test images: %f %%' % (total,
                                                                    accuracy))
    print('Finished Adversarial')

    return accuracy, adversarial_succ


def simple_eps_noise(testloader, net, eps):
    """
    Implementation of noise addition in epsilon strength.
    :return:
    """
    adversarial_succ = []
    classes = ('car', 'truck', 'cat')
    class_correct = list(0. for i in range(3))
    class_total = list(0. for i in range(3))
    correct_total = 0
    curr_label = 0
    k = 0
    for data in testloader:
        images, labels = data
        images.requires_grad = True
        outputs = net(images)
        sm = torch.nn.Softmax(dim=1)
        probabilities = sm(outputs)
        _, predicted = torch.max(probabilities.data, 1)
        correct = predicted.item() == labels.item()
        if not correct:
            continue
        loss = F.nll_loss(outputs, labels)
        net.zero_grad()
        loss.backward()
        grads = images.grad.data
        noised_im = apply_noise_single_im(images, eps, grads)
        new_output = net(noised_im)
        probabilities = sm(new_output)
        _, new_predicted = torch.max(probabilities.data, 1)
        correct = new_predicted.item() == labels.item()
        class_total[labels] += 1
        k += 1
        if not correct:
            if len(adversarial_succ) < 5 and curr_label == predicted.item():
                noised_ex = noised_im.squeeze().detach().cpu().numpy()
                adversarial_succ.append((predicted.item(), new_predicted.item(), noised_ex))
                curr_label = (curr_label + 1) % 3
        else:
            correct_total += 1
            class_correct[new_predicted] += 1

    avg = 0
    for i in range(3):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))
        avg += 100 * class_correct[i] / class_total[i]

    avg /= 3
    total = sum(class_total)
    accuracy = 100 * correct_total / total
    print('Average Accuracy of the network on the %d test images: %f %%' % (total, avg))
    print('Accuracy of the network on the %d test images: %f %%' % (total,
            accuracy))
    print('Finished Adversarial')

    return accuracy, adversarial_succ


def adversarial(weights_path):
    """
    Function runs the epsilon method
    """
    epsilon = [0.05, 0.1]
    results = []
    for i in range(len(epsilon)):
        net = SimpleModel()
        net.load(weights_path)
        net.eval()
        trans = get_transform()
        testset = torchvision.datasets.ImageFolder(root='./data/test_data', transform=trans)
        # data = dataset.get_dataset_as_torch_dataset('./data/dev.pickle')
        testloader = DataLoader(testset, batch_size=1, shuffle=True, num_workers=0)
        results.append( simple_eps_noise(testloader, net, epsilon[i])[1] )

    show_examples(results, epsilon)


def optimized_adver(weights_path):
    """
    Runs the optimize method
    """
    results = []
    epsilon = [0]
    net = SimpleModel()
    net.load(weights_path)
    net.eval()
    trans = get_transform(False)
    testset = torchvision.datasets.ImageFolder(root='./data/test_data', transform=trans)
    # data = dataset.get_dataset_as_torch_dataset('./data/dev.pickle')
    testloader = DataLoader(testset, batch_size=1, shuffle=True, num_workers=0)
    noise = torch.rand(3, 32, 32)
    results.append( optimize_noise(testloader, net, noise, 0.0001)[1])

    show_examples(results, epsilon)


optimized_adver(weights_path='./data/my_trained_model14.ckpt')
# adversarial(weights_path='./data/my_trained_model14.ckpt')
