# code structure inspired by https://github.com/dyhan0920/PyramidNet-PyTorch/blob/master/train.py

import argparse
import copy
import json
import os
import shutil
import time
import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import tqdm

from arg_extractor import get_args
from data_providers import K49Dataset
from model_architectures import AdjustLayer, AlexCifarNet, LeNet, ResNet

warnings.filterwarnings("ignore")


def main():
    global args, best_err1, device, num_classes, num_out_feats
    args = get_args()
    torch.manual_seed(args.random_seed)

    # most cases have 10 classes
    # if there are more, then it will be reassigned
    num_classes = 10
    best_err1 = 100

    # define datasets
    if args.target == "mnist":
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_set_all = datasets.MNIST(
            'data', train=True, transform=transform_train, target_transform=None, download=True)
        # set aside 10000 examples from the training set for validation
        train_set, val_set = torch.utils.data.random_split(
            train_set_all, [50000, 10000])
        # if we do experiments with variable target set size, this will take care of it
        # by default the target set size is 50000
        target_set_size = min(50000, args.target_set_size)
        train_set, _ = torch.utils.data.random_split(
            train_set, [target_set_size, 50000 - target_set_size])
        test_set = datasets.MNIST(
            'data', train=False, transform=transform_test, target_transform=None, download=True)
    elif args.target == "kmnist":
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_set_all = datasets.KMNIST(
            'data', train=True, transform=transform_train, target_transform=None, download=True)
        # set aside 10000 examples from the training set for validation
        train_set, val_set = torch.utils.data.random_split(
            train_set_all, [50000, 10000])
        target_set_size = min(50000, args.target_set_size)
        # if we do experiments with variable target set size, this will take care of it
        train_set, _ = torch.utils.data.random_split(
            train_set, [target_set_size, 50000 - target_set_size])
        test_set = datasets.KMNIST(
            'data', train=False, transform=transform_test, target_transform=None, download=True)
    elif args.target == "k49":
        num_classes = 49
        train_images = np.load('./data/k49-train-imgs.npz')['arr_0']
        test_images = np.load('./data/k49-test-imgs.npz')['arr_0']
        train_labels = np.load('./data/k49-train-labels.npz')['arr_0']
        test_labels = np.load('./data/k49-test-labels.npz')['arr_0']

        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        # set aside about 10% of training data for validation
        train_set_all = K49Dataset(
            train_images, train_labels, transform=transform_train)
        train_set, val_set = torch.utils.data.random_split(
            train_set_all, [209128, 23237])
        # currently we do not support variable target set size for k49
        # enable this to use it
        # target_set_size = min(209128, args.target_set_size)
        # train_set, _ = torch.utils.data.random_split(
        #     train_set, [target_set_size, 209128 - target_set_size])
        test_set = K49Dataset(
            test_images, test_labels, transform=transform_test)
    elif args.target == "cifar10":
        normalize = transforms.Normalize(
            mean=[x / 255.0 for x in [125.3, 123.0, 113.9]], std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        transform_train = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            normalize
        ])

        transform_test = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            normalize
        ])

        train_set_all = datasets.CIFAR10(
            'data', train=True, transform=transform_train, target_transform=None, download=True)
        # set aside 5000 examples from the training set for validation
        train_set, val_set = torch.utils.data.random_split(
            train_set_all, [45000, 5000])
        # if we do experiments with variable target set size, this will take care of it
        target_set_size = min(45000, args.target_set_size)
        train_set, _ = torch.utils.data.random_split(
            train_set, [target_set_size, 45000 - target_set_size])
        test_set = datasets.CIFAR10(
            'data', train=False, transform=transform_test, target_transform=None, download=True)
    elif args.target == "cifar100":
        num_classes = 100
        normalize = transforms.Normalize(
            mean=[x / 255.0 for x in [125.3, 123.0, 113.9]], std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        transform_train = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            normalize
        ])

        transform_test = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            normalize
        ])

        train_set_all = datasets.CIFAR100(
            'data', train=True, transform=transform_train, target_transform=None, download=True)
        # set aside 5000 examples from the training set for validation
        train_set, val_set = torch.utils.data.random_split(
            train_set_all, [45000, 5000])
        # if we do experiments with variable target set size, this will take care of it
        target_set_size = min(45000, args.target_set_size)
        train_set, _ = torch.utils.data.random_split(
            train_set, [target_set_size, 45000 - target_set_size])
        test_set = datasets.CIFAR100(
            'data', train=False, transform=transform_test, target_transform=None, download=True)

    # create data loaders
    if args.baseline:
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.num_base_examples, shuffle=True)
    else:
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False)

    # create data loaders to get base examples
    if args.source == "emnist":
        train_set_source = datasets.EMNIST(
            'data', 'letters', train=True, download=True, transform=transform_train, target_transform=None)
        train_loader_source = torch.utils.data.DataLoader(
            train_set_source, batch_size=args.num_base_examples, shuffle=True)
    elif args.source == "mnist":
        train_set_source = datasets.MNIST(
            'data', train=True, download=True, transform=transform_train, target_transform=None)
        train_loader_source = torch.utils.data.DataLoader(
            train_set_source, batch_size=args.num_base_examples, shuffle=True)
    elif args.source == "kmnist":
        train_set_source = datasets.KMNIST(
            'data', train=True, download=True, transform=transform_train, target_transform=None)
        train_loader_source = torch.utils.data.DataLoader(
            train_set_source, batch_size=args.num_base_examples, shuffle=True)
    elif args.source == "cifar10":
        train_set_source = datasets.CIFAR10(
            'data', train=True, download=True, transform=transform_train, target_transform=None)
        train_loader_source = torch.utils.data.DataLoader(
            train_set_source, batch_size=args.num_base_examples, shuffle=True)
    elif args.source == "cifar100":
        train_set_source = datasets.CIFAR100(
            'data', train=True, download=True, transform=transform_train, target_transform=None)
        train_loader_source = torch.utils.data.DataLoader(
            train_set_source, batch_size=args.num_base_examples, shuffle=True)
    elif args.source == "svhn":
        train_set_source = datasets.SVHN(
            'data', split='train', download=True, transform=transform_train, target_transform=None)
        train_loader_source = torch.utils.data.DataLoader(
            train_set_source, batch_size=args.num_base_examples, shuffle=True)
    elif args.source == "cub":
        # modify the root depending on where you place the images
        cub_data_root = './data/CUB_200_2011/images'
        train_set_source = datasets.ImageFolder(
            cub_data_root, transform=transform_train, target_transform=None)
        train_loader_source = torch.utils.data.DataLoader(
            train_set_source, batch_size=args.num_base_examples, shuffle=True)
    elif args.source == "fake":
        # there is also an option to use random noise base examples
        if args.target == "mnist":
            num_channels = 1
            dims = 28
        else:
            num_channels = 3
            dims = 32
        train_set_source = datasets.FakeData(size=5000, image_size=(
            num_channels, dims, dims), num_classes=10, transform=transform_train, target_transform=None, random_offset=0)
        train_loader_source = torch.utils.data.DataLoader(
            train_set_source, batch_size=args.num_base_examples, shuffle=True)
    else:
        # get the fixed images from the same dataset as the training data
        train_set_source = train_set
        train_loader_source = torch.utils.data.DataLoader(
            train_set, batch_size=args.num_base_examples, shuffle=True)

    if torch.cuda.is_available():  # checks whether a cuda gpu is available
        device = torch.cuda.current_device()

        print("use GPU", device)
        print("GPU ID {}".format(torch.cuda.current_device()))
    else:
        print("use CPU")
        device = torch.device('cpu')  # sets the device to be CPU

    train_loader_source_iter = iter(train_loader_source)

    if args.balanced_source:
        # use a balanced set of fixed examples - same number of examples per class
        class_counts = {}
        fixed_input = []
        fixed_target = []

        for batch_fixed_i, batch_fixed_t in train_loader_source_iter:
            if sum(class_counts.values()) >= args.num_base_examples:
                break
            for fixed_i, fixed_t in zip(batch_fixed_i, batch_fixed_t):
                if len(class_counts.keys()) < num_classes:
                    if int(fixed_t) in class_counts:
                        if class_counts[int(fixed_t)] < args.num_base_examples // num_classes:
                            class_counts[int(fixed_t)] += 1
                            fixed_input.append(fixed_i)
                            fixed_target.append(int(fixed_t))
                    else:
                        class_counts[int(int(fixed_t))] = 1
                        fixed_input.append(fixed_i)
                        fixed_target.append(int(fixed_t))
                else:
                    if int(fixed_t) in class_counts:
                        if class_counts[int(fixed_t)] < args.num_base_examples // num_classes:
                            class_counts[int(fixed_t)] += 1
                            fixed_input.append(fixed_i)
                            fixed_target.append(int(fixed_t))
        fixed_input = torch.stack(fixed_input).to(device=device)
        fixed_target = torch.Tensor(fixed_target).to(device=device)
    else:
        # used for cross-dataset scenario - random selection of classes
        # not taking into accound the original classes
        fixed_input, fixed_target = next(train_loader_source_iter)
        fixed_input = fixed_input.to(device=device)
        fixed_target = fixed_target.to(device=device)

    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss().to(device=device)

    # start at uniform labels and then learn them
    labels = torch.zeros((args.num_base_examples, num_classes),
                         requires_grad=True, device=device)
    labels = labels.new_tensor([[float(1.0 / num_classes) for e in range(num_classes)] for i in range(
        args.num_base_examples)], requires_grad=True, device=device)

    # define an optimizer for labels
    labels_opt = torch.optim.Adam([labels])
    cudnn.benchmark = True

    # define the models to use
    if args.target == "cifar10" or args.target == "cifar100":
        if args.resnet:
            model = ResNet(dataset=args.target, depth=18, num_classes=num_classes,
                           bottleneck=False).to(device=device)
            num_out_feats = 64
            model_name = 'resnet'
        else:
            model = AlexCifarNet(args).to(device=device)
            num_out_feats = 192
            model_name = 'alexnet'
    else:
        model = LeNet(args).to(device=device)
        num_out_feats = 84
        model_name = 'LeNet'
    feat_opt = torch.optim.Adam(model.parameters())

    # define a layer to scale the classifier weights (global RR classifier weights)
    # calibration inspired by https://github.com/bertinetto/r2d2 and https://github.com/kjunelee/MetaOptNet
    adjust = AdjustLayer(device=device)
    # use Adam optimizer for learning the calibration
    # use a higher learning rate to learn the calibration quickly
    # since it is only learned using a small number of examples / steps
    # we do not want to learn it using target set data
    adj_opt = torch.optim.Adam(list(adjust.parameters()), 0.01)

    # define initial global classifier weights
    classifier_weights = torch.zeros(
        num_out_feats + 1, num_classes, device=device, requires_grad=False)

    if args.baseline:
        create_json_experiment_log(fixed_target)
        # remap the targets - only relevant in cross-dataset
        fixed_target = remap_targets(fixed_target, num_classes)
        # printing the labels helps ensure the seeds work
        print('The labels of the fixed examples are')
        print(fixed_target.tolist())
        labels = one_hot(fixed_target.long(), num_classes)

        # add smoothing to the baseline if selected
        if args.label_smoothing > 0:
            labels = create_smooth_labels(
                labels, args.label_smoothing, num_classes)

        # use the validation set to find a suitable number of iterations for training
        num_baseline_steps, errors_list, num_steps_used = find_best_num_steps(
            val_loader, criterion, fixed_input, labels)
        print('Number of steps to use for the baseline: ' + str(num_baseline_steps))
        experiment_update_dict = {'num_baseline_steps': num_baseline_steps,
                                  'errors_list': errors_list,
                                  'num_steps_used': num_steps_used}
        update_json_experiment_log_dict(experiment_update_dict)

        if args.test_various_models:
            assert args.target == "cifar10"
            model_name_list = ['alexnet', 'LeNet', 'resnet']

            for model_name_test in model_name_list:
                # do 20 repetitions of training from scratch
                for test_i in range(20):
                    print('Test repetition ' + str(test_i))
                    test_err1, test_loss = test(
                        test_loader, model_name_test, criterion, fixed_input, labels, num_baseline_steps)
                    print('Test error (top-1 error):', test_err1)
                    experiment_update_dict = {'test_top_1_error_' + model_name_test: test_err1,
                                              'test_loss_' + model_name_test: test_loss,
                                              'num_test_steps_' + model_name_test: num_baseline_steps}
                    update_json_experiment_log_dict(experiment_update_dict)
        else:
            # do 20 repetitions of training from scratch
            for test_i in range(20):
                print('Test repetition ' + str(test_i))
                test_err1, test_loss = test(
                    test_loader, model_name, criterion, fixed_input, labels, num_baseline_steps)
                print('Test error (top-1 error):', test_err1)
                experiment_update_dict = {'test_top_1_error': test_err1,
                                          'test_loss': test_loss,
                                          'num_test_steps': num_baseline_steps}
                update_json_experiment_log_dict(experiment_update_dict)
    else:
        create_json_experiment_log(fixed_target)

        # start measuring time
        start_time = time.time()

        # initialize variables to decide when to restart a model
        ma_list = []
        ma_sum = 0
        lowest_ma_sum = 999999999
        current_num_steps = 0
        num_steps_list = []
        num_steps_from_min = 0

        val_err1 = 100.0
        val_loss = 5.0
        num_steps_val = 0

        with tqdm.tqdm(total=args.epochs) as pbar_epochs:
            for epoch in range(0, args.epochs):
                train_err1, train_loss, labels, meta_loss, classifier_weights, feat_loss, ma_list, ma_sum, lowest_ma_sum, current_num_steps, num_steps_list, num_steps_from_min, model, feat_opt, adj_opt, adjust = \
                    train(train_loader, model, fixed_input, labels, criterion,
                          labels_opt, epoch, classifier_weights, feat_opt,
                          ma_list, ma_sum, lowest_ma_sum, current_num_steps, num_steps_list, num_steps_from_min, adj_opt, adjust)
                # evaluate on the validation set only every 5 epochs as it can be quite expensive to train a new model from scratch
                if epoch % 5 == 4:
                    # calculate the number of steps to use
                    if len(num_steps_list) == 0:
                        num_steps_val = current_num_steps
                    else:
                        num_steps_val = int(np.mean(num_steps_list[-3:]))

                    val_err1, val_loss = validate(
                        val_loader, model, criterion, epoch, fixed_input, labels, num_steps_val)

                    if val_err1 <= best_err1:
                        best_labels = labels.detach().clone()
                        best_num_steps = num_steps_val
                        best_err1 = min(val_err1, best_err1)

                    print('Current best val error (top-1 error):', best_err1)

                pbar_epochs.update(1)

                experiment_update_dict = {'train_top_1_error': train_err1,
                                          'train_loss': train_loss,
                                          'val_top_1_error': val_err1,
                                          'val_loss': val_loss,
                                          'feat_loss': feat_loss,
                                          'epoch': epoch,
                                          'meta_loss': meta_loss,
                                          'num_val_steps': num_steps_val}
                # save the best labels so that we can analyse them
                if epoch == args.epochs - 1:
                    experiment_update_dict['labels'] = best_labels.tolist()

                update_json_experiment_log_dict(experiment_update_dict)

        print('Best val error (top-1 error):', best_err1)

        # stop measuring time
        experiment_update_dict = {'total_train_time': time.time() - start_time}
        update_json_experiment_log_dict(experiment_update_dict)

        # this does number of steps analysis - what happens if we do more or fewer steps for training
        if args.num_steps_analysis:
            num_steps_add = [-50, -20, -10, 0, 10, 20, 50, 100]

            for num_steps_add_item in num_steps_add:
                # start measuring time for testing
                start_time = time.time()
                local_errs = []
                local_losses = []
                local_num_steps = best_num_steps + num_steps_add_item
                print('Number of steps for training: ' + str(local_num_steps))
                # each number of steps will have a robust estimate by using 20 repetitions
                for test_i in range(20):
                    print('Test repetition ' + str(test_i))
                    test_err1, test_loss = test(
                        test_loader, model_name, criterion, fixed_input, best_labels, local_num_steps)
                    local_errs.append(test_err1)
                    local_losses.append(test_loss)
                    print('Test error (top-1 error):', test_err1)
                experiment_update_dict = {'test_top_1_error': local_errs,
                                          'test_loss': local_losses,
                                          'total_test_time': time.time() - start_time,
                                          'num_test_steps': local_num_steps}
                update_json_experiment_log_dict(experiment_update_dict)
        else:
            if args.test_various_models:
                assert args.target == "cifar10", "test various models is only meant to be used for CIFAR-10"
                model_name_list = ['alexnet', 'LeNet', 'resnet']

                for model_name_test in model_name_list:
                    for test_i in range(20):
                        print(model_name_test)
                        print('Test repetition ' + str(test_i))
                        test_err1, test_loss = test(
                            test_loader, model_name_test, criterion, fixed_input, best_labels, best_num_steps)
                        print('Test error (top-1 error):', test_err1)
                        experiment_update_dict = {'test_top_1_error_' + model_name_test: test_err1,
                                                  'test_loss_' + model_name_test: test_loss,
                                                  'total_test_time_' + model_name_test: time.time() - start_time,
                                                  'num_test_steps_' + model_name_test: best_num_steps}
                        update_json_experiment_log_dict(experiment_update_dict)
            else:
                for test_i in range(20):
                    print(model_name)
                    print('Test repetition ' + str(test_i))
                    test_err1, test_loss = test(
                        test_loader, model_name, criterion, fixed_input, best_labels, best_num_steps)
                    print('Test error (top-1 error):', test_err1)
                    experiment_update_dict = {'test_top_1_error': test_err1,
                                              'test_loss': test_loss,
                                              'total_test_time': time.time() - start_time,
                                              'num_test_steps': best_num_steps}
                    update_json_experiment_log_dict(experiment_update_dict)


def one_hot(indices, depth):
    """
    Returns a one-hot tensor.
    This is a PyTorch equivalent of Tensorflow's tf.one_hot.

    Parameters:
      indices:  a (n_batch, m) Tensor or (m) Tensor.
      depth: a scalar. Represents the depth of the one hot dimension.
    Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
    """
    encoded_indicies = torch.zeros(
        indices.size() + torch.Size([depth])).to(device=device)
    index = indices.view(indices.size() + torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1, index, 1)

    return encoded_indicies


def create_smooth_labels(labels, label_smoothing, num_classes):
    labels = labels * (1.0 - label_smoothing)
    labels = labels + label_smoothing / num_classes
    return labels


def remap_targets(fixed_target, depth):
    """Only useful for cross-dataset scenarios - within dataset it does not remap it"""
    if int(max(fixed_target)) < depth:
        if args.source == "cifar10" or args.source == "cifar100" or args.source == "svhn":
            return fixed_target
        else:
            return fixed_target.long()
    else:
        remapped_targets = []
        mapping = {}
        max_index = 0
        for label in fixed_target:
            if int(label) in mapping:
                remapped_targets.append(mapping[int(label)])
            else:
                mapping[int(label)] = max_index
                max_index += 1
                remapped_targets.append(mapping[int(label)])
        if max_index > depth:
            raise('Too many labels to remap')
        return torch.Tensor(remapped_targets).long().to(device=device)


def train(train_loader, model, fixed_input, labels, criterion, labels_opt, epoch, classifier_weights, feat_opt,
          ma_list, ma_sum, lowest_ma_sum, current_num_steps, num_steps_list, num_steps_from_min, adj_opt, adjust):
    """
    Do one epoch of training the synthetic labels.

    Parameters ma_list, ma_sum, lowest_ma_sum, current_num_steps, num_steps_list, num_steps_from_min
    are used for keeping track of statistics for model resets across epochs.
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    feat_losses = AverageMeter()
    top1 = AverageMeter()
    meta_loss_meter = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    w = classifier_weights.detach().clone()

    # define over how many steps to calculate the moving average
    # for resetting the model and also how many steps to wait
    # we use the same value for both
    if args.target == "cifar100" or args.target == "k49":
        stats_gap = 100
    elif args.num_base_examples > 100:
        stats_gap = 200
    else:
        stats_gap = 50

    for i, (input_, target) in enumerate(train_loader):
        # sample a minibatch of n_o target dataset examples x_t' with labels y_t'

        # measure data loading time
        data_time.update(time.time() - end)

        input_ = input_.to(device=device)
        target = target.to(device=device)

        # sample a minibatch of n_i examples from x~, y~: x~', y~'
        perm = torch.randperm(fixed_input.size(0))
        idx = perm[:args.inner_batch_size]
        fi_i = fixed_input[idx]
        lb_i = labels[idx]

        # solve the problem to find local weights w_l that minimize L(f_w_l(f_theta (x~')), y~')$
        feature = model(fi_i)
        feature_add_bias = torch.cat(
            (feature, torch.ones((args.inner_batch_size, 1), device=device)), 1)
        w_local = get_local_weights(feature_add_bias, lb_i, device)

        # update w <-- (1 - alpha) w + w_l (pseudo-gradient)
        w = (1 - 0.01) * w.detach() + 0.01 * w_local

        # find the logits for target data - using calibrated global weights
        logit = model(input_, adjust() * w)
        label_loss = criterion(logit, target)
        meta_loss_meter.update(label_loss.item(), input_.size(0))
        # update y~ <-- y~ - beta nabla_y~ L(f_w(f_theta(x_t)), y_t)

        labels_opt.zero_grad()
        label_loss.backward(retain_graph=True)
        labels_opt.step()

        # normalize the labels to form a valid probability distribution
        labels.data = torch.clamp(labels.data, 0, 1)
        labels.data = labels.data / labels.data.sum(dim=1).unsqueeze(1)

        # now update the model - using calibrated global weights
        # update theta <-- theta - beta nabla_theta L(f_w(f_theta(x~')), y~')
        feat_output = model(fi_i, adjust() * w)
        # ignore gradients from the labels
        feat_loss = soft_cross_entropy(feat_output, labels[idx].detach())

        # update the feature weights and also the calibration parameter
        feat_opt.zero_grad()
        adj_opt.zero_grad()
        feat_loss.backward()
        feat_opt.step()
        adj_opt.step()

        feat_losses.update(feat_loss.item(), input_.size(0))

        # measure accuracy and record loss
        err1 = compute_error_rate(logit.data, target, topk=(1,))[0]  # it returns a list

        losses.update(label_loss.item(), input_.size(0))
        top1.update(err1.item(), input_.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # update the moving average statistics
        if len(ma_list) < stats_gap:
            ma_list.append(err1.item())
            ma_sum += err1.item()
            current_num_steps += 1

            current_ma = ma_sum / len(ma_list)

            if current_num_steps == stats_gap:
                lowest_ma_sum = ma_sum
                num_steps_from_min = 0
        else:
            ma_sum = ma_sum - ma_list[0] + err1.item()
            ma_list = ma_list[1:] + [err1.item()]
            current_num_steps += 1

            current_ma = ma_sum / len(ma_list)

            if ma_sum < lowest_ma_sum:
                lowest_ma_sum = ma_sum
                num_steps_from_min = 0
            elif num_steps_from_min < stats_gap:
                num_steps_from_min += 1
            else:
                # do early stopping
                num_steps_list.append(
                    current_num_steps - num_steps_from_min - 1)
                # restart all metrics
                ma_list = []
                ma_sum = 0
                lowest_ma_sum = 999999999
                current_num_steps = 0
                num_steps_from_min = 0

                # restart the model and the optimizers
                if args.target == "cifar10" or args.target == "cifar100":
                    if args.resnet:
                        model = ResNet(dataset=args.target, depth=18, num_classes=num_classes,
                                       bottleneck=False).to(device=device)
                    else:
                        model = AlexCifarNet(args).to(device=device)
                else:
                    model = LeNet(args).to(device=device)
                feat_opt = torch.optim.Adam(model.parameters())
                adjust = AdjustLayer(device=device)
                adj_opt = torch.optim.Adam(list(adjust.parameters()), 0.01)

                w = torch.zeros(
                    num_out_feats + 1, num_classes, device=device, requires_grad=False)

                print('Model restarted after ' +
                      str(num_steps_list[-1]) + ' steps')

        if i % args.print_freq == 0 and args.verbose is True:
            print('Epoch: [{0}/{1}][{2}/{3}]\t'
                  'LR: {LR:.6f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1-err {top1.val:.4f} ({top1.avg:.4f})\t'
                  'MAvg 1-err {current_ma:.4f}'.format(
                      epoch, args.epochs, i, len(train_loader), LR=1, batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1, current_ma=current_ma))

    print('* Epoch: [{0}/{1}]\t Top 1-err {top1.avg:.3f}  Train Loss {loss.avg:.3f}'.format(
        epoch, args.epochs, top1=top1, loss=losses))

    return top1.avg, losses.avg, labels, meta_loss_meter.avg, w, feat_losses.avg, \
        ma_list, ma_sum, lowest_ma_sum, current_num_steps, num_steps_list, num_steps_from_min, \
        model, feat_opt, adj_opt, adjust


def soft_cross_entropy(pred, soft_targets):
    """A method for calculating cross entropy with soft targets"""
    logsoftmax = nn.LogSoftmax()
    return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))

def validate(val_loader, model, criterion, epoch, fixed_input, labels, num_steps):
    """Validation phase. This involves training a model from scratch."""
    losses_eval = AverageMeter()
    top1_eval = AverageMeter()
    start = time.time()

    print('Number of steps for retraining during validation: ' + str(num_steps))

    # initialize a new model
    if args.target == "cifar10" or args.target == "cifar100":
        if args.resnet:
            model_eval = ResNet(dataset=args.target, depth=18, num_classes=num_classes,
                                bottleneck=False).to(device=device)
        else:
            model_eval = AlexCifarNet(args).to(device=device)
    else:
        model_eval = LeNet(args).to(device=device)
    feat_opt = torch.optim.Adam(model_eval.parameters())
    adjust = AdjustLayer(device=device)
    adj_opt = torch.optim.Adam(list(adjust.parameters()), 0.01)
    model_eval.train()

    w_eval = torch.zeros(num_out_feats + 1, num_classes,
                         device=device, requires_grad=False)

    # train a model from scratch for the given number of steps
    # using only the base examples and their synthetic labels
    for ITER in range(num_steps):
        # sample a minibatch of n_i examples from x~, y~: x~', y~'
        perm = torch.randperm(fixed_input.size(0))
        idx = perm[:args.inner_batch_size]
        fi_i = fixed_input[idx]
        lb_i = labels[idx]

        # solve the problem to find local weights w_l that minimize L(f_w_l(f_theta (x~')), y~')$
        feature = model_eval(fi_i)
        feature_add_bias = torch.cat(
            (feature, torch.ones((args.inner_batch_size, 1), device=device)), 1)
        w_local = get_local_weights(feature_add_bias, lb_i, device)
        # update w <-- (1 - alpha) w + alpha w_l (pseudo-gradient)
        w_eval = (1 - 0.01) * w_eval.detach() + 0.01 * w_local

        # update theta <-- theta - beta nabla_theta L(f_w(f_theta(x~')), y~')
        # also update the calibration
        feat_output = model_eval(fi_i, adjust() * w_eval)
        feat_loss = soft_cross_entropy(feat_output, labels[idx].detach())
        feat_opt.zero_grad()
        adj_opt.zero_grad()
        feat_loss.backward()
        feat_opt.step()
        adj_opt.step()

    # evaluate the validation error
    # switch to evaluate mode
    model_eval.eval()

    for i, (input_, target) in enumerate(val_loader):
        input_ = input_.to(device=device)
        target = target.to(device=device)
        output = model_eval(input_, w_eval)
        loss = criterion(output, target)

        # measure error and record loss
        err1 = compute_error_rate(output.data, target, topk=(1,))[0]
        top1_eval.update(err1.item(), input_.size(0))
        losses_eval.update(loss.item(), input_.size(0))

    val_time = time.time() - start
    print('* Epoch: [{0}/{1}]\t Top 1-err {top1.avg:.3f} Val Loss {loss.avg:.3f} Time {val_time:.3f}'.format(
        epoch, args.epochs, top1=top1_eval, loss=losses_eval, val_time=val_time))

    return top1_eval.avg, losses_eval.avg


def test(test_loader, model_name, criterion, fixed_input, labels, num_steps):
    """Test phase. This involves training a model from scratch."""
    losses_eval = AverageMeter()
    top1_eval = AverageMeter()
    start = time.time()

    print('Number of steps for retraining during test: ' + str(num_steps))

    # initialize a new model
    if args.target == "cifar10" or args.target == "cifar100":
        if args.test_various_models:
            if model_name == 'resnet':
                model_eval = ResNet(dataset=args.target, depth=18, num_classes=num_classes,
                                    bottleneck=False).to(device=device)
            elif model_name == 'LeNet':
                model_eval = LeNet(args).to(device=device)
            else:
                model_eval = AlexCifarNet(args).to(device=device)
        else:
            if args.resnet:
                model_eval = ResNet(dataset=args.target, depth=18, num_classes=num_classes,
                                    bottleneck=False).to(device=device)
            else:
                model_eval = AlexCifarNet(args).to(device=device)
    else:
        model_eval = LeNet(args).to(device=device)
    model_eval.train()

    if args.test_various_models:
        if model_name == 'resnet':
            w_eval = torch.zeros(
                64 + 1, num_classes, device=device, requires_grad=False)
        elif model_name == 'LeNet':
            w_eval = torch.zeros(
                84 + 1, num_classes, device=device, requires_grad=False)
        else:
            w_eval = torch.zeros(
                192 + 1, num_classes, device=device, requires_grad=False)
    else:
        w_eval = torch.zeros(
            num_out_feats + 1, num_classes, device=device, requires_grad=False)

    feat_opt = torch.optim.Adam(model_eval.parameters())
    adjust = AdjustLayer(device=device)
    adj_opt = torch.optim.Adam(list(adjust.parameters()), 0.01)

    # train a model from scratch for the given number of steps
    # using only the base examples and their synthetic labels
    for ITER in range(num_steps):
        # sample a minibatch of n_i examples from x~, y~: x~', y~'
        perm = torch.randperm(fixed_input.size(0))
        idx = perm[:args.inner_batch_size]
        fi_i = fixed_input[idx]
        lb_i = labels[idx]

        # solve the problem to find local weights w_l that minimize L(f_w_l(f_theta (x~')), y~')$
        feature = model_eval(fi_i)
        feature_add_bias = torch.cat(
            (feature, torch.ones((args.inner_batch_size, 1), device=device)), 1)
        w_local = get_local_weights(feature_add_bias, lb_i, device)

        # update w <-- alpha w + (1-alpha) w_l (pseudo-gradient)
        w_eval = (1 - 0.01) * w_eval.detach() + 0.01 * w_local
        feat_output = model_eval(fi_i, adjust() * w_eval)

        feat_loss = soft_cross_entropy(feat_output, labels[idx].detach())
        feat_opt.zero_grad()
        adj_opt.zero_grad()
        feat_loss.backward()
        feat_opt.step()
        adj_opt.step()

    # evaluate the test error
    # switch to evaluate mode
    model_eval.eval()

    for i, (input_, target) in enumerate(test_loader):
        input_ = input_.to(device=device)
        target = target.to(device=device)
        output = model_eval(input_, w_eval)
        loss = criterion(output, target)

        # measure error and record loss
        err1 = compute_error_rate(output.data, target, topk=(1,))[0]
        top1_eval.update(err1.item(), input_.size(0))
        losses_eval.update(loss.item(), input_.size(0))

    test_time = time.time() - start
    print('Testing with the feature extractor and global SVM parameters trained from scratch')
    print('Test time: ' + str(test_time))
    print(
        'Test error (top-1 error): {top1_eval.avg:.4f}'.format(top1_eval=top1_eval))

    return top1_eval.avg, losses_eval.avg


def find_best_num_steps(val_loader, criterion, fixed_input, labels):
    """Calculate the best number of steps to use based on the validation set."""
    best_num_steps = 0
    lowest_err = 100
    errors_list = []
    num_steps_used = []

    # use a larger number of max steps when using more base examples
    if args.num_base_examples > 100:
        max_num_steps = 1701
    else:
        max_num_steps = 1000

    # initialize a new model
    if args.target == "cifar10" or args.target == "cifar100":
        if args.resnet:
            model_eval = ResNet(dataset=args.target, depth=18, num_classes=num_classes,
                                bottleneck=False).to(device=device)
        else:
            model_eval = AlexCifarNet(args).to(device=device)
    else:
        model_eval = LeNet(args).to(device=device)
    feat_opt = torch.optim.Adam(model_eval.parameters())
    adjust = AdjustLayer(device=device)
    adj_opt = torch.optim.Adam(list(adjust.parameters()), 0.01)
    model_eval.train()

    w_eval = torch.zeros(num_out_feats + 1, num_classes,
                         device=device, requires_grad=False)

    for ITER in range(max_num_steps):
        # sample a minibatch of n_i examples from x~, y~: x~', y~'
        perm = torch.randperm(fixed_input.size(0))
        idx = perm[:args.inner_batch_size]
        fi_i = fixed_input[idx]
        lb_i = labels[idx]

        feature = model_eval(fi_i)
        feature_add_bias = torch.cat(
            (feature, torch.ones((args.inner_batch_size, 1), device=device)), 1)
        w_local = get_local_weights(feature_add_bias, lb_i, device)
        # update w <-- alpha w + (1-alpha) w_l (pseudo-gradient)
        w_eval = (1 - 0.01) * w_eval.detach() + 0.01 * w_local

        # update theta <-- theta - beta nabla_theta L(f_w(f_theta(x~')), y~')
        feat_output = model_eval(fi_i, adjust() * w_eval)

        feat_loss = soft_cross_entropy(feat_output, labels[idx].detach())
        feat_opt.zero_grad()
        adj_opt.zero_grad()
        feat_loss.backward()
        feat_opt.step()
        adj_opt.step()

        # for larger numbers of base examples we decrease the frequency of evaluation
        if args.num_base_examples > 100:
            if ITER in set([9, 24, 74, 100, 200, 300, 500, 700, 1000, 1200, 1500, 1700]):
                validate_now = True
            else:
                validate_now = False
        else:
            if ITER % 50 == 49 or ITER in set([9, 24, 74]):
                validate_now = True
            else:
                validate_now = False

        if validate_now:
            # switch to evaluate mode
            model_eval.eval()
            top1_eval = AverageMeter()
            losses_eval = AverageMeter()

            for i, (input_, target) in enumerate(val_loader):
                input_ = input_.to(device=device)
                target = target.to(device=device)
                output = model_eval(input_, w_eval)
                loss = criterion(output, target)

                # measure error and record loss
                err1 = compute_error_rate(output.data, target, topk=(1,))[0]
                top1_eval.update(err1.item(), input_.size(0))
                losses_eval.update(loss.item(), input_.size(0))

            errors_list.append(top1_eval.avg)
            num_steps_used.append(ITER + 1)
            if top1_eval.avg < lowest_err:
                lowest_err = top1_eval.avg
                best_num_steps = ITER + 1

    print(num_steps_used)
    print(errors_list)
    return best_num_steps, errors_list, num_steps_used


def create_json_experiment_log(fixed_target):
    json_experiment_log_file_name = os.path.join(
        'results', args.expname) + '.json'
    experiment_summary_dict = {'train_top_1_error': [], 'train_loss': [],
                               'val_top_1_error': [], 'val_loss': [],
                               'test_top_1_error': [], 'test_loss': [],
                               'epoch': [], 'labels': [], 'meta_loss': [],
                               'total_train_time': [], 'total_test_time': [], 'feat_loss': [],
                               'num_val_steps': [], 'num_test_steps': [],
                               'base_example_labels': fixed_target.tolist()}
    with open(json_experiment_log_file_name, 'w') as f:
        json.dump(experiment_summary_dict, fp=f)


def update_json_experiment_log_dict(experiment_update_dict):
    json_experiment_log_file_name = os.path.join(
        'results', args.expname) + '.json'
    with open(json_experiment_log_file_name, 'r') as f:
        summary_dict = json.load(fp=f)

    for key in experiment_update_dict:
        if key not in summary_dict:
            summary_dict[key] = []
        summary_dict[key].append(experiment_update_dict[key])

    with open(json_experiment_log_file_name, 'w') as f:
        json.dump(summary_dict, fp=f)


def compute_error_rate(output, target, topk=(1,)):
    """Computes the error rate"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        wrong_k = batch_size - correct_k
        res.append(wrong_k.mul_(100.0 / batch_size))

    return res


def get_local_weights(feature_add_bias, labels, device):
    """
    A method to calculate ridge regression weights for a minibatch.

    Parameters:
    feature_add_bias: features
    labels: soft labels for the features
    device: the device where the parameters are placed

    Return:
    ridge_sol: the weights that solve the ridge regression problem.
    """

    # value of the parameter for L2 regularization
    # we tried various values: e.g. 0.01, 0.1, 1.0, 10
    # 1.0 worked the best on the validation set
    l2_regularizer_lambda = 1.0
    # construct a kernel matrix to decrease the dimensionality of the problem
    # linear kernel - but other kernels could be tried too
    kernel_matrix = feature_add_bias.matmul(feature_add_bias.transpose(0, 1))
    id_matrix = torch.eye(args.inner_batch_size).to(device=device)

    # define the objective of the ridge regression problem
    ridge_obj = kernel_matrix + l2_regularizer_lambda * id_matrix
    # find the inverse of the objective as the first part of solution
    ridge_sol = torch.inverse(ridge_obj)
    # include features and labels to obtain the actual solution
    ridge_sol = feature_add_bias.transpose(0, 1).matmul(ridge_sol)
    ridge_sol = ridge_sol.matmul(labels)
    return ridge_sol


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()
