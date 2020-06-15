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
from model_architectures import AdjustLayer, AlexCifarNet, LeNet, ResNet

warnings.filterwarnings("ignore")


def main():
    global args, best_err1, device, num_classes, num_out_feats
    args = get_args()
    torch.manual_seed(args.random_seed)
    best_err1 = 100

    # define datasets
    if args.target == "mnist":
        normalize = transforms.Normalize((0.1307,), (0.3081,))
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        train_set_all = datasets.MNIST(
            'data', train=True, transform=transform_train, target_transform=None, download=True)
        # set aside 10000 examples from the training set for validation
        train_set, val_set = torch.utils.data.random_split(
            train_set_all, [50000, 10000])
        # if we do experiments with variable target set size, this will take care of it
        target_set_size = min(50000, args.target_set_size)
        train_set, _ = torch.utils.data.random_split(
            train_set, [target_set_size, 50000 - target_set_size])
        test_set = datasets.MNIST(
            'data', train=False, transform=transform_test, target_transform=None, download=True)
        num_classes = 10
        num_channels = 1
        input_size = 28
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
        num_classes = 10
        num_channels = 3
        input_size = 32
    else:
        raise "The dataset is not currently supported"

    # create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False)

    if torch.cuda.is_available():  # checks whether a cuda gpu is available
        device = torch.cuda.current_device()
        print("use GPU", device)
        print("GPU ID {}".format(torch.cuda.current_device()))
    else:
        print("use CPU")
        device = torch.device('cpu')  # sets the device to be CPU

    # randomly initualize the images and create associated images
    # labels [0, 1, 2, ..., 0, 1, 2, ...]
    distill_labels = torch.arange(num_classes, dtype=torch.long, device=device) \
        .repeat(args.num_base_examples // num_classes, 1).reshape(-1)
    distill_labels = one_hot(distill_labels, num_classes)
    distill_data = torch.rand(args.num_base_examples, num_channels, input_size, input_size,
                              device=device, requires_grad=True)
    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss().to(device=device)
    # define an optimizer for learning the synthetic images
    data_opt = torch.optim.Adam([distill_data])

    # define a layer to scale the classifier weights (global RR classifier weights)
    # calibration inspired by https://github.com/bertinetto/r2d2 and https://github.com/kjunelee/MetaOptNet
    adjust = AdjustLayer(device=device)
    # use Adam optimizer for learning the calibration
    # use a higher learning rate to learn the calibration quickly
    # since it is only learned using a small number of examples / steps
    # we do not want to learn it using target set data
    adj_opt = torch.optim.Adam(list(adjust.parameters()), 0.01)
    cudnn.benchmark = True

    # define the models to use
    if args.target == "cifar10":
        model = AlexCifarNet(args).to(device=device)
    else:
        model = LeNet(args).to(device=device)
    feat_opt = torch.optim.Adam(model.parameters())

    if args.target == "cifar10":
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

    # define initial global classifier weights
    classifier_weights = torch.zeros(
        num_out_feats + 1, num_classes, device=device, requires_grad=False)

    create_json_experiment_log()

    # start measuring time
    start_time = time.time()

    # initialize early stopping variables
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
            train_err1, train_loss, distill_data, meta_loss, classifier_weights, feat_loss, ma_list, ma_sum, lowest_ma_sum, current_num_steps, num_steps_list, num_steps_from_min, model, feat_opt, adj_opt, adjust = \
                train(train_loader, model, distill_data, distill_labels, criterion,
                      data_opt, epoch, classifier_weights, feat_opt,
                      ma_list, ma_sum, lowest_ma_sum, current_num_steps, num_steps_list, num_steps_from_min, adj_opt, adjust, normalize)
            # evaluate on the validation set only every 5 epochs as it can be quite expensive to train a new model from scratch
            if epoch % 5 == 4:
                # calculate the number of steps to use
                if len(num_steps_list) == 0:
                    num_steps_val = current_num_steps
                else:
                    num_steps_val = int(np.mean(num_steps_list[-3:]))

                val_err1, val_loss = validate(
                    val_loader, model, criterion, epoch, distill_data, distill_labels, num_steps_val, normalize)
                # otherwise the stats keep the previous value

                if val_err1 <= best_err1:
                    best_distill_data = distill_data.detach().clone()
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
            # save the best images so that we can analyse them
            if epoch == args.epochs - 1:
                experiment_update_dict['data'] = best_distill_data.tolist()

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
                    test_loader, model, criterion, best_distill_data, distill_labels, local_num_steps, normalize)
                local_errs.append(test_err1)
                local_losses.append(test_loss)
                print('Test error (top-1 error):', test_err1)
            experiment_update_dict = {'test_top_1_error': local_errs,
                                      'test_loss': local_losses,
                                      'total_test_time': time.time() - start_time,
                                      'num_test_steps': local_num_steps}
            update_json_experiment_log_dict(experiment_update_dict)
    else:
        # evaluate on test set repeatedly - always do 20 repetitions
        for test_i in range(20):
            print('Test repetition ' + str(test_i))
            test_err1, test_loss = test(
                test_loader, model, criterion, best_distill_data, distill_labels, best_num_steps, normalize)

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


def train(train_loader, model, distill_data, distill_labels, criterion, data_opt, epoch, classifier_weights, feat_opt,
          ma_list, ma_sum, lowest_ma_sum, current_num_steps, num_steps_list, num_steps_from_min, adj_opt, adjust, normalize):
    """
    Do one epoch of training the synthetic images.

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
    stats_gap = 50

    for i, (input_, target) in enumerate(train_loader):
        # sample a minibatch of n_o target dataset examples x_t' with labels y_t'

        # measure data loading time
        data_time.update(time.time() - end)

        input_ = input_.to(device=device)
        target = target.to(device=device)

        # sample a minibatch of n_i examples from x~, y~: x~', y~'
        perm = torch.randperm(distill_data.size(0))
        idx = perm[:args.inner_batch_size]
        # normalize the synthetic images using the standard normalization
        fi_i = torch.stack([normalize(image)
                            for image in distill_data[idx].cpu()]).to(device=device)
        lb_i = distill_labels[idx]

        # solve the problem to find local weights w_l that minimize L(f_w_l(f_theta (x~')), y~')$
        feature = model(fi_i)
        feature_add_bias = torch.cat(
            (feature, torch.ones((args.inner_batch_size, 1), device=device)), 1)
        w_local = get_local_weights(feature_add_bias, lb_i, device)

        # update w <-- (1 - alpha) w + w_l (pseudo-gradient)
        w = (1 - 0.01) * w.detach() + 0.01 * w_local

        # find the logits for target data - using calibrated global weights
        logit = model(input_, adjust() * w)
        data_loss = criterion(logit, target)
        meta_loss_meter.update(data_loss.item(), input_.size(0))
        # update x~ <-- x~ - beta nabla_x~ L(f_w(f_theta(x_t)), y_t)
        data_opt.zero_grad()
        data_loss.backward(retain_graph=True)
        data_opt.step()

        # make sure the synthetic examples have valid values after the update
        distill_data.data = torch.clamp(distill_data.data, 0, 1)

        # update the feature weights and also the calibration parameter
        # update theta <-- theta - beta nabla_theta L(f_w(f_theta(x~')), y~')
        feat_output = model(fi_i.detach(), adjust() * w)
        feat_loss = soft_cross_entropy(feat_output, lb_i)
        feat_opt.zero_grad()
        adj_opt.zero_grad()
        feat_loss.backward()
        feat_opt.step()
        adj_opt.step()

        feat_losses.update(feat_loss.item(), input_.size(0))

        # measure error rate and record loss
        err1 = compute_error_rate(logit.data, target, topk=(1,))[0]  # it returns a list

        losses.update(data_loss.item(), input_.size(0))
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

                # restart the model and the optimizer
                if args.target == "cifar10":
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

    return top1.avg, losses.avg, distill_data, meta_loss_meter.avg, w, feat_losses.avg, \
        ma_list, ma_sum, lowest_ma_sum, current_num_steps, num_steps_list, num_steps_from_min, \
        model, feat_opt, adj_opt, adjust


def soft_cross_entropy(pred, soft_targets):
    """A method for calculating cross entropy with soft targets"""
    logsoftmax = nn.LogSoftmax()
    return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))


def validate(val_loader, model, criterion, epoch, distill_data, distill_labels, num_steps, normalize):
    """Validation phase. This involves training a model from scratch."""
    losses_eval = AverageMeter()
    top1_eval = AverageMeter()
    start = time.time()

    print('Number of steps for retraining during validation: ' + str(num_steps))

    # initialize a new model
    if args.target == "cifar10":
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
    # using only the synthetic images and their labels
    for ITER in range(num_steps):
        # sample a minibatch of n_i examples from x~, y~: x~', y~'
        perm = torch.randperm(distill_data.size(0))
        idx = perm[:args.inner_batch_size]
        fi_i = torch.stack([normalize(image)
                            for image in distill_data[idx].detach().cpu()]).to(device=device)
        lb_i = distill_labels[idx]

        # solve the problem to find local weights w_l that minimize L(f_w_l(f_theta (x~')), y~')$
        feature = model_eval(fi_i)
        feature_add_bias = torch.cat(
            (feature, torch.ones((args.inner_batch_size, 1), device=device)), 1)
        w_local = get_local_weights(feature_add_bias, lb_i, device)

        # update w <-- (1 - alpha) w + alpha w_l (pseudo-gradient)
        w_eval = (1 - 0.01) * w_eval.detach() + 0.01 * w_local

        # update theta <-- theta - beta nabla_theta L(f_w(f_theta(x~')), y~')
        # also update the calibration
        feat_output = model_eval(fi_i.detach(), adjust() * w_eval)
        feat_loss = soft_cross_entropy(feat_output, lb_i)
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

        # measure error rate and record loss
        err1 = compute_error_rate(output.data, target, topk=(1,))[0]
        top1_eval.update(err1.item(), input_.size(0))
        losses_eval.update(loss.item(), input_.size(0))

    val_time = time.time() - start
    print('* Epoch: [{0}/{1}]\t Top 1-err {top1.avg:.3f} Val Loss {loss.avg:.3f} Time {val_time:.3f}'.format(
        epoch, args.epochs, top1=top1_eval, loss=losses_eval, val_time=val_time))

    return top1_eval.avg, losses_eval.avg


def test(test_loader, model, criterion, distill_data, distill_labels, num_steps, normalize):
    """Test phase. This involves training a model from scratch."""
    losses_eval = AverageMeter()
    top1_eval = AverageMeter()
    start = time.time()

    print('Number of steps for retraining during test: ' + str(num_steps))

    # initialize a new model
    if args.target == "cifar10":
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
    # using only the synthetic images and their labels
    for ITER in range(num_steps):
        # sample a minibatch of n_i examples from x~, y~: x~', y~'
        perm = torch.randperm(distill_data.size(0))
        idx = perm[:args.inner_batch_size]
        fi_i = torch.stack([normalize(image)
                            for image in distill_data[idx].detach().cpu()]).to(device=device)
        lb_i = distill_labels[idx]

        feature = model_eval(fi_i)
        feature_add_bias = torch.cat(
            (feature, torch.ones((args.inner_batch_size, 1), device=device)), 1)
        w_local = get_local_weights(feature_add_bias, lb_i, device)

        w_eval = (1 - 0.01) * w_eval.detach() + 0.01 * w_local

        # update theta <-- theta - beta nabla_theta L(f_w(f_theta(x~')), y~')
        feat_output = model_eval(fi_i.detach(), adjust() * w_eval)

        feat_loss = soft_cross_entropy(feat_output, lb_i)
        feat_opt.zero_grad()
        adj_opt.zero_grad()
        feat_loss.backward()
        feat_opt.step()
        adj_opt.step()

    # evaluate the validation error
    # switch to evaluate mode
    model_eval.eval()

    for i, (input_, target) in enumerate(test_loader):
        input_ = input_.to(device=device)
        target = target.to(device=device)
        output = model_eval(input_, w_eval)
        loss = criterion(output, target)

        # measure error rate and record loss
        err1 = compute_error_rate(output.data, target, topk=(1,))[0]
        top1_eval.update(err1.item(), input_.size(0))
        losses_eval.update(loss.item(), input_.size(0))

    test_time = time.time() - start
    print('Testing with the feature extractor and global SVM parameters trained from scratch')
    print('Test time: ' + str(test_time))
    print(
        'Test error (top-1 error): {top1_eval.avg:.4f}'.format(top1_eval=top1_eval))

    return top1_eval.avg, losses_eval.avg


def create_json_experiment_log():
    json_experiment_log_file_name = os.path.join(
        'results', args.expname) + '.json'
    experiment_summary_dict = {'train_top_1_error': [], 'train_loss': [],
                               'val_top_1_error': [], 'val_loss': [],
                               'test_top_1_error': [], 'test_loss': [],
                               'epoch': [], 'labels': [], 'meta_loss': [],
                               'total_train_time': [], 'total_test_time': [], 'feat_loss': [],
                               'num_val_steps': [], 'num_test_steps': []}
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
    # 1.0 worked the best on the validation set - for label distillation
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
