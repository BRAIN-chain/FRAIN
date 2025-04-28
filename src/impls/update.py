#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn
# from torchvision.datasets import CIFAR10
# from torch.utils.data import DataLoader, Dataset
from torch.utils.data import Dataset

from airbench.dataloader import CifarLoader
from airbench.model import train


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

        self.classes = self.dataset.classes
        self.data = dataset.data[self.idxs]
        self.targets = [dataset.targets[i] for i in self.idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        # return (image).clone().detach(), torch.tensor(label)
        return image, label


class LocalUpdate(object):
    def __init__(self, args, hyps, dataset, idxs, logger, gpu=0):
        self.args = args
        self.hyps = hyps
        self.logger = logger
        self.device = torch.device(gpu)

        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            dataset, list(idxs))

        # Default criterion set to NLL loss function
        # self.criterion = nn.NLLLoss().to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(0.9*len(idxs))]
        # idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]

        trainloader = CifarLoader(DatasetSplit(dataset, idxs_train),
                                  train=True, batch_size=self.args.local_bs, aug={'flip': True, 'translate': 2, })
        # validloader = CifarLoader(DatasetSplit(dataset, idxs_val),
        #  batch_size=self.args.local_bs)
        testloader = CifarLoader(DatasetSplit(dataset, idxs_test),
                                 train=False, batch_size=2000)
        return trainloader, None, testloader

    def update_weights(self, model, epochs=9.9, global_round=None):
        # Set Params
        batch_size = self.hyps['opt']['batch_size']
        momentum = self.hyps['opt']['momentum']
        # Assuming gradients are constant in time, for Nesterov momentum, the below ratio is how much
        # larger the default steps will be than the underlying per-example gradients. We divide the
        # learning rate by this ratio in order to ensure steps are the same scale as gradients, regardless
        # of the choice of momentum.
        kilostep_scale = 1024 * (1 + 1 / (1 - momentum))
        # un-decoupled learning rate for PyTorch SGD
        # lr = self.hyps['opt']['lr'] / kilostep_scale
        lr = self.args.lr / kilostep_scale
        wd = self.hyps['opt']['weight_decay'] * batch_size / kilostep_scale
        lr_biases = lr * self.hyps['opt']['bias_scaler']
        label_smoothing = self.hyps['opt']['label_smoothing']
        whiten_bias_epochs = self.hyps['opt']['whiten_bias_epochs']

        tta_level = self.hyps['net']['tta_level']

        train_acc_collect, train_loss_collect, acc_collect = train(
            # run,
            model,
            self.trainloader, self.testloader,
            batch_size, epochs, momentum, lr, wd, lr_biases,
            label_smoothing, whiten_bias_epochs,
            tta_level
        )

        return model.state_dict(), sum(train_loss_collect) / len(train_loss_collect)

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        loss /= total
        return accuracy, loss


class ByzantineLocalUpdate(LocalUpdate):
    def update_weights(self, model, epochs=9.9, global_round=None):
        w_t = copy.deepcopy(model.state_dict())
        for key in w_t.keys():
            w_t[key] = torch.zeros_like(w_t[key])
        return w_t, None


def test_inference(args, model, test_dataset, gpu=0):
    """ Returns the test accuracy and loss.
    """
    device = torch.device(gpu)

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    # criterion = nn.NLLLoss().to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    testloader = CifarLoader(
        test_dataset, batch_size=args.local_bs)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    loss /= total
    return accuracy, loss
