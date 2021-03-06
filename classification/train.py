#*
# @file Different utility functions
# Copyright (c) Yaohui Cai, Zhewei Yao, Zhen Dong, Amir Gholami
# All rights reserved.
# This file is part of ZeroQ repository.
#
# ZeroQ is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ZeroQ is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ZeroQ repository.  If not, see <http://www.gnu.org/licenses/>.
#*

import argparse
import torch
import numpy as np
import torch.nn as nn
from pytorchcv.model_provider import get_model as ptcv_get_model
from utils import *
from distill_data import *
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

# model settings
def arg_parse():
    parser = argparse.ArgumentParser(
        description='This repository contains the PyTorch implementation for the paper ZeroQ: A Novel Zero-Shot Quantization Framework.')
    parser.add_argument('--dataset',
                        type=str,
                        default='imagenet',
                        choices=['imagenet', 'cifar10'],
                        help='type of dataset')
    parser.add_argument('--model',
                        type=str,
                        default='mobilenetv2_w1',
                        choices=[
                            'resnet18', 'resnet50', 'inceptionv3',
                            'mobilenetv2_w1', 'shufflenet_g1_w1',
                            'resnet20_cifar10', 'resnet56_cifar10',
                            'resnext29_32x4d_cifar10', 'sqnxt23_w2'
                        ],
                        help='model to be quantized')
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='batch size of distilled data')
    parser.add_argument('--test_batch_size',
                        type=int,
                        default=256,
                        help='batch size of test data')

    parser.add_argument('--train_batch_size',
                        type=int,
                        default=256,
                        help='batch size of train data')
    parser.add_argument('--epochs',
                        type=int,
                        default=10,
                        help="Training epochs")
    parser.add_argument('--weight_decay',
                        type=float,
                        default=1e-5)
    parser.add_argument('--learning_rate',
                        type=float,
                        default=5e-4)
    parser.add_argument('--momentum',
                        type=float,
                        default=0.9)
    parser.add_argument("--lr-decay",
                        type=float,
                        default=0.1)
    parser.add_argument("--save",
                        type=str,
                        default="model_path")
    parser.add_argument("--eval-interval",
                        type=int,
                        default=1,
                        help="the evaluation interval during training")
    parser.add_argument("--init-test",
                        type=bool,
                        default=0,
                        help="whether test the initial result w/o QAT")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = arg_parse()
    print(args)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # Load pretrained model
    model = ptcv_get_model(args.model, pretrained=True).cuda()
    print('****** Full precision model loaded ******')

    # Load training data
    train_loader = getTrainData(args.dataset,
                              batch_size=args.train_batch_size,
                              path='./data/imagenet12/',
                              for_inception=args.model.startswith('inception'))
    # Load validation data
    test_loader = getTestData(args.dataset,
                              batch_size=args.test_batch_size,
                              path='./data/imagenet12/',
                              for_inception=args.model.startswith('inception'))

    print('****** Data loaded ******')

    if args.dataset == "cifar10":
        criterion_smooth = CrossEntropyLabelSmooth(10, 0.1).cuda()
    elif args.dataset == "imagenet":
        criterion_smooth = CrossEntropyLabelSmooth(1000, 0.1).cuda()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer,
        milestones=[3, 5, 8],
        gamma=args.lr_decay
    )

    args.loss_function = criterion_smooth
    args.optimizer     = optimizer
    args.scheduler     = scheduler

    quantized_model = quantize_model(model, first_layer=True)
    quantized_model = nn.DataParallel(quantized_model).cuda()
    quantized_model.train()

    best_acc = 0
    if args.init_test:
        acc = test(model, test_loader)

    for epoch in range(args.epochs):
        acc, loss = train(quantized_model, train_loader, args, test_loader)
        print(f"Epoch {epoch}: loss = {loss:4f}, top1_accuracy = {acc*100:4f}, learning_rate = {args.scheduler.get_last_lr()[0]}")
        if (epoch+1) % args.eval_interval == 0:
            acc = test(quantized_model, test_loader)

            if acc > best_acc:
                best_acc = acc
                save_path = os.path.join(args.dataset, args.save)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                filename = os.path.join( save_path, "bestmodel.pth.tar")
                torch.save({'state_dict': model.state_dict(),}, filename)


