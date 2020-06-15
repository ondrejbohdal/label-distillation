import argparse
import json
import os
import sys


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    """
    Returns a namedtuple with arguments extracted from the command line.
    :return: A namedtuple with arguments
    """

    parser = argparse.ArgumentParser(
        description='Label Distillation')
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--batch_size', default=1024, type=int,
                        metavar='N', help='outer minibatch size')
    parser.add_argument('--inner_batch_size', default=50, type=int,
                        metavar='N', help='inner minibatch size')
    parser.add_argument('--num_base_examples', default=100, type=int,
                        metavar='N', help='total number of base examples')
    parser.add_argument('--print-freq', default=1, type=int,
                        metavar='N', help='print frequency')
    parser.add_argument('--no-verbose', dest='verbose', action='store_false',
                        help='to print the status at every iteration')
    parser.add_argument('--expname', default='short_experiment', type=str,
                        help='name of experiment')
    parser.add_argument('--source', default='emnist', type=str,
                        help='from which dataset to take base examples')
    parser.add_argument('--target', default='mnist', type=str,
                        help='from which dataset to take target examples')
    parser.add_argument('--balanced_source', default=False, type=str2bool,
                        help='if to use class-balanced source with the same number of classes')
    parser.add_argument('--baseline', default=False, type=str2bool,
                        help='if to use baseline approach')
    parser.add_argument('--resnet', default=False, type=str2bool,
                        help='if to use a resnet feature extractor')
    parser.add_argument('--num_steps_analysis', default=False, type=str2bool,
                        help='if to conduct analysis of variable number of steps for testing')
    parser.add_argument('--test_various_models', default=False, type=str2bool,
                        help='if to test with various models')
    parser.add_argument('--target_set_size', default=50000, type=int,
                        metavar='N', help='how many examples to use from target set for training')
    parser.add_argument('--random_seed', default=1234, type=int,
                        metavar='N', help='the random seed to use')
    parser.add_argument('--label_smoothing', default=0, type=float,
                        help='the value of label smoothing for baseline training')
    parser.add_argument('--inner_lr', default=0.01, type=float,
                        metavar='N', help='the inner loop learning rate')
    parser.add_argument('--filepath_to_arguments_json_file', nargs="?", type=str, default=None,
                        help='')

    args = parser.parse_args()

    if args.filepath_to_arguments_json_file is not None:
        args = extract_args_from_json(
            json_file_path=args.filepath_to_arguments_json_file, existing_args_dict=args)

    assert args.inner_batch_size <= args.num_base_examples, "inner batch size should not be larger than the number of base examples"

    arg_str = [(str(key), str(value)) for (key, value) in vars(args).items()]
    print(arg_str)

    return args


class AttributeAccessibleDict(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


def extract_args_from_json(json_file_path, existing_args_dict=None):
    summary_filename = json_file_path
    with open(summary_filename) as f:
        arguments_dict = json.load(fp=f)

    for key, value in vars(existing_args_dict).items():
        if key not in arguments_dict:
            arguments_dict[key] = value

    arguments_dict = AttributeAccessibleDict(arguments_dict)

    return arguments_dict
