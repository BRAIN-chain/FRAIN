#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import time
import pickle
from tqdm import tqdm

from impls.options import args_parser
from impls.update import test_inference
from impls.utils import get_dataset, exp_details

from airbench.dataloader import CifarLoader
from airbench.model import make_net, train
from airbench.hyperparameters import hyp


if __name__ == '__main__':
    start_time = time.time()

    args = args_parser()
    exp_details(args)

    # load datasets
    train_dataset, test_dataset, _ = get_dataset(args)

    # BUILD MODEL
    if (args.model == 'cnn') and (args.dataset == 'cifar'):
        pass
    else:
        exit('Error: unrecognized model')
    # Make Model
    widths = hyp['net']['widths']
    batchnorm_momentum = hyp['net']['batchnorm_momentum']
    scaling_factor = hyp['net']['scaling_factor']
    global_model = make_net(widths, batchnorm_momentum, scaling_factor)
    global_model.train()
    print(global_model)

    # Training
    trainloader = CifarLoader(train_dataset,
                              train=True, batch_size=args.local_bs, aug={'flip': True, 'translate': 2, })
    testloader = CifarLoader(test_dataset,
                             train=False, batch_size=2000)

    # Set Params
    batch_size = hyp['opt']['batch_size']
    # epochs = hyp['opt']['train_epochs']
    epochs = args.epochs
    momentum = hyp['opt']['momentum']
    # Assuming gradients are constant in time, for Nesterov momentum, the below ratio is how much
    # larger the default steps will be than the underlying per-example gradients. We divide the
    # learning rate by this ratio in order to ensure steps are the same scale as gradients, regardless
    # of the choice of momentum.
    kilostep_scale = 1024 * (1 + 1 / (1 - momentum))
    # un-decoupled learning rate for PyTorch SGD
    # lr = hyp['opt']['lr'] / kilostep_scale
    lr = args.lr / kilostep_scale
    wd = hyp['opt']['weight_decay'] * batch_size / kilostep_scale
    lr_biases = lr * hyp['opt']['bias_scaler']
    label_smoothing = hyp['opt']['label_smoothing']
    whiten_bias_epochs = hyp['opt']['whiten_bias_epochs']

    tta_level = hyp['net']['tta_level']

    train_acc_collect, train_loss_collect, test_acc_collect = train(
        # run,
        global_model,
        trainloader, testloader,
        batch_size, epochs, momentum, lr, wd, lr_biases,
        label_smoothing, whiten_bias_epochs,
        tta_level
    )

    # testing
    test_acc, test_loss = test_inference(args, global_model, test_dataset)
    print(f'test acc: {test_acc}, test_loss: {test_loss}')

    # Saving the objects:
    file_name = './save/objects/nn_{}_{}_{}_{}.pkl'.\
        format(args.dataset, args.model, args.epochs, time.time())
    with open(file_name, 'wb') as f:
        pickle.dump([train_loss_collect, test_acc_collect], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    # PLOTTING (optional)
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')

    # Plot Loss curve
    plt.figure()
    plt.title('Training Loss vs Communication rounds')
    plt.plot(range(len(train_loss_collect)), train_loss_collect, color='r')
    plt.ylabel('Training loss')
    plt.xlabel('Communication Rounds')
    plt.savefig('./save/nn_{}_{}_train_loss.png'.format(args.dataset, args.model,
                                                        args.epochs))

    # Plot Average Accuracy vs Communication rounds
    plt.figure()
    plt.title('Average Accuracy vs Communication rounds (Train)')
    plt.plot(range(len(train_acc_collect)), test_acc_collect, color='k')
    plt.ylabel('Average Accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig('./save/nn_{}_{}_train_acc.png'.format(args.dataset, args.model,
                                                       args.epochs))

    # Plot Average Accuracy vs Communication rounds
    plt.figure()
    plt.title('Average Accuracy vs Communication rounds (Test)')
    plt.plot(range(len(test_acc_collect)), test_acc_collect, color='k')
    plt.ylabel('Average Accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig('./save/nn_{}_{}_test_acc.png'.format(args.dataset, args.model,
                                                      args.epochs))
