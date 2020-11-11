# train.py
#!/usr/bin/env	python3

""" train network using pytorch

author baiyu
"""

import os
import sys
import argparse
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR

import math
import sys
sys.path.append('..')
import resnet
from polynomial_scheduler import PolyLR

from constant_sum_v2 import MadamV2,SGDV2,AdamV2
from constant_sum_v3 import CSV3
from constant_sum_v4 import CSV4
from constant_sum_v5 import CSV5
from constant_sum_v6 import CSV6
from constant_sum_v7 import CSV7,SGDV7,FromageCSV7
from constant_sum_v8 import CSV8,MadamV8
from fromage import Fromage
from madam import Madam
def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes) 
    return y[labels]
def train(epoch):
    pf = True #False
    start = time.time()
    net.train()

    for batch_index, (images, labels) in enumerate(cifar100_training_loader):
        if epoch <= args.warm:
            warmup_scheduler.step()

        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()

        optimizer.zero_grad()
        outputs = net(images) * args.os # scaled output
        
        if args.loss == "mse":
            if args.task == 'cifar100':
                num_classes = 100
            else:
                num_classes = 10
            y = torch.eye(num_classes) 
            one_hot = y[labels].cuda()
            loss = loss_function(outputs, one_hot)
        else:
            loss = loss_function(outputs, labels)
        if pf:
            # print("output mean: {}, var: {}".format(outputs.mean(dim=0).mean(),outputs.var(dim=0).mean()))
            # print("target mean: {}, var: {}".format(one_hot.mean(dim=0).mean(),one_hot.var(dim=0).mean()))
            # print("output mean: {}, var: {}, dim :{}".format(outputs.mean(dim=0),outputs.var(dim=0),outputs.mean(dim=0).shape))
            # print("target mean: {}, var: {}".format(one_hot.mean(dim=0),one_hot.var(dim=0)))
            print(outputs.shape)
            pf = False
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(cifar100_training_loader) + batch_index + 1

        last_layer = list(net.children())[-1]
        for name, para in last_layer.named_parameters():
            if 'weight' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
            if 'bias' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.b + len(images),
            total_samples=len(cifar100_training_loader.dataset)
        ))

        #update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)
    #optimizer.clean_weight()
    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))

@torch.no_grad()
def eval_training(dataloader=None, train=False, epoch=None):

    start = time.time()
    net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0

    for (images, labels) in dataloader:

        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images) * args.os # scaled output
        if args.loss == "mse":
            if args.task == 'cifar100':
                num_classes = 100
            else:
                num_classes = 10
            y = torch.eye(num_classes) 
            one_hot = y[labels].cuda()
            loss = loss_function(outputs, one_hot)
        else:
            loss = loss_function(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()
    if args.gpu:
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')
    print('Evaluating Network.....')
    if train:
        name = "train"
    else:
        name = "test"
    print('{} set: Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        name,
        test_loss / len(dataloader.dataset),
        correct.float() / len(dataloader.dataset),
        finish - start
    ))
    print()

    #add informations to tensorboard
    if train:
        writer.add_scalar('Train/Average loss', test_loss / len(dataloader.dataset), epoch)
        writer.add_scalar('Train/Accuracy', correct.float() / len(dataloader.dataset), epoch)
    else:
        writer.add_scalar('Test/Average loss', test_loss / len(dataloader.dataset), epoch)
        writer.add_scalar('Test/Accuracy', correct.float() / len(dataloader.dataset), epoch)

    return correct.float() / len(dataloader.dataset)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    #Yang
    parser.add_argument('--init_bias', action='store_true')

    parser.add_argument('--res', action='store_true')
    parser.add_argument('--bn', action='store_true')
    parser.add_argument('--na', action='store_true')
    parser.add_argument('--bias', action='store_true')
    parser.add_argument('--beta', default=0.999, type=float,help='beta for csv4')
    parser.add_argument('--clip', default=1.0, type=float)
    parser.add_argument('--sp', default=0.0, type=float)
    parser.add_argument('--a', default=1.0 , type=float)
    parser.add_argument('--scale', default=1.0 , type=float)
    parser.add_argument('--os', default=1.0, type=float)
    parser.add_argument('--w_mul', default=1.0 , type=float)
    parser.add_argument('--nl', default='relu', type=str)
    parser.add_argument('--prefix', default='', type=str)
    parser.add_argument('--gamma', default=0.2 , type=float)
    parser.add_argument('--lr-exp', default=1.0 , type=float)
    parser.add_argument('--wd', default=0.0005 , type=float)
    parser.add_argument('--momentum', default=0.9 , type=float)
    parser.add_argument('--free', action='store_false', dest="fix_norm", default=True)

    parser.add_argument('--wb', action='store_true', help="weight buffer for csv7")
    parser.add_argument('--wn', default=0.0 , type=float, help="weight noise for csv7")
    parser.add_argument('--nlr', action='store_true', help="noise scale with lr for csv7")

    parser.add_argument('--grad', action='store_true', help="gradual learning")

    parser.add_argument('--alpha', default=0.0 , type=float,help='alpha for noise')

    parser.add_argument('--loss', default='ce', type=str,help="mse or crossentropy (ce) loss," )
    parser.add_argument('--optimizer', default='sgd', type=str)
    parser.add_argument('--lr-decay-epoch', default='40,80', type=str)
    parser.add_argument('--sch', default='step', type=str)
    parser.add_argument('--lr-pow', default=6.0 , type=float)
    parser.add_argument('--task', default='cifar100' , type=str)
    
    parser.add_argument('--no-aug', action='store_false', dest="da", default=True)

    args = parser.parse_args()

    net = get_network(args)

    weights = []
    for i in range(100):
        w = [1] * (i + 1)
        w.extend([0] * (99 - i))
        weights.append(w)
    #print(weights)
    weights = torch.FloatTensor(weights).cuda()
    #data preprocessing:
    if args.task == "cifar100":
        mean = settings.CIFAR100_TRAIN_MEAN
        std = settings.CIFAR100_TRAIN_STD
    elif args.task == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    else:
        print("invalid task!!")
    cifar100_training_loader = get_training_dataloader(
        mean,
        std,
        num_workers=4,
        batch_size=args.b,
        shuffle=True,
        alpha = args.alpha,
        task = args.task,
        da = args.da
    )

    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=False,
        task = args.task
    )
    #test training acc
    cifar100_train_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=False,
        task = args.task,
        train = True
    )

    settings.MILESTONES = [120,150,180]
    if args.loss == "ce":
        if args.grad:
            loss_function = nn.CrossEntropyLoss(weight=weights[0])
        else:
            loss_function = nn.CrossEntropyLoss()
            print("weights: {}".format(weights[99]))
    elif args.loss == "mse":
        print("using mse loss!!")
        loss_function = nn.MSELoss()
        args.prefix += args.loss
    
    if args.optimizer == 'sgd':
        print("using sgd!")
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
        #settings.MILESTONES = [120,150,180]
    elif args.optimizer == 'csv4':
        print("using csv4!")
        if args.clip == 0:
            clip = math.inf
        else:
            clip = abs(args.clip   )
        #settings.MILESTONES = [120,150,180]
        optimizer = CSV4(net.parameters(), lr=args.lr, beta=args.beta, bias_clip=clip)
    elif args.optimizer == 'csv5':
        print("using csv5!")
        if args.clip == 0:
            clip = math.inf
        else:
            clip = abs(args.clip   )
        #settings.MILESTONES = [120,150,180]
        optimizer = CSV5(net.parameters(), lr=args.lr, beta1=args.momentum, beta2=args.beta, bias_clip=clip)
    elif args.optimizer == 'csv6':
        print("using csv6!")
        if args.clip == 0:
            clip = math.inf
        else:
            clip = abs(args.clip   )
        #settings.MILESTONES = [120,150,180]
        optimizer = CSV6(net.parameters(), lr=args.lr, beta1=args.momentum, beta2=args.beta, bias_clip=clip, weight_decay=args.wd)
    elif args.optimizer == 'csv7':
        print("using csv7!")
        if args.clip == 0:
            clip = math.inf
        else:
            clip = abs(args.clip   )
        #settings.MILESTONES = [120,150,180]
        optimizer = CSV7(net.parameters(),scale=args.scale, lr=args.lr, beta1=args.momentum, beta2=args.beta, bias_clip=clip, weight_decay=args.wd,
                            noise=args.wn, weight_buffer=args.wb,noise_with_lr=args.nlr)
    elif args.optimizer == 'sgdv7':
        print("using sgdv7!")
        if args.clip == 0:
            clip = math.inf
        else:
            clip = abs(args.clip   )
        #settings.MILESTONES = [120,150,180]
        optimizer = SGDV7(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd,bias_clip=clip,
                            noise=args.wn, weight_buffer=args.wb,noise_with_lr=args.nlr)
    
    elif args.optimizer == 'csv8':
        print("using csv8!")
        if args.clip == 0:
            clip = math.inf
        else:
            clip = abs(args.clip   )
        #settings.MILESTONES = [120,150,180]
        optimizer = CSV8(net.parameters(),scale=args.scale, lr=args.lr, beta1=args.momentum, beta2=args.beta, bias_clip=clip, weight_decay=args.wd,
                            noise=args.wn, weight_buffer=args.wb,noise_with_lr=args.nlr,fix_norm=args.fix_norm,lr_exp_base=args.lr_exp)

    elif args.optimizer == 'madamv8':
        print("using madamv8!")
        if args.clip == 0:
            clip = math.inf
        else:
            clip = abs(args.clip   )
        #settings.MILESTONES = [120,150,180]
        optimizer = MadamV8(net.parameters(), lr=args.lr,beta1=args.momentum, beta2=args.beta,bias_clip=clip,
                            flip_thr=args.wn,fix_norm=args.fix_norm)
        # wn is used for flip threshold
    elif args.optimizer == 'fromagecsv7':
        print("using fromagecsv7!")
        if args.clip == 0:
            clip = math.inf
        else:
            clip = abs(args.clip   )
        #settings.MILESTONES = [120,150,180]
        optimizer = FromageCSV7(net.parameters(), lr=args.lr, weight_decay=args.wd,bias_clip=clip,
                                                beta1=args.momentum, beta2=args.beta)
    elif args.optimizer == 'fromage':
        print("using fromage!")
        if args.clip == 0:
            clip = math.inf
        else:
            clip = abs(args.clip   )
        #settings.MILESTONES = [120,150,180]
        optimizer = Fromage(net.parameters(), lr=args.lr)
    elif args.optimizer == 'madam':
        print("using madam!")
        if args.clip == 0:
            clip = math.inf
        else:
            clip = abs(args.clip   )
        #settings.MILESTONES = [120,150,180]
        optimizer = Madam(net.parameters(), lr=args.lr)

    if args.sch == "step":
        train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=args.gamma) #learning rate decay
    elif args.sch == "poly":
        train_scheduler = PolyLR(optimizer,T_max=settings.EPOCH, eta_min=0, power=args.lr_pow)
        args.prefix += 'pow' + str(args.lr_pow) 
    elif args.sch == "cos":
        train_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings.EPOCH) #learning rate decay
        args.prefix += 'cos'  
    iter_per_epoch = len(cifar100_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)
    if not args.alpha == 0 or True:
        args.prefix += '_noise' + str(args.alpha) + '_'
    args.prefix = args.task + args.prefix + '_scale'+ str(args.scale) +"_os" + str(args.os)
    args.prefix += 'free_norm_' if not args.fix_norm else ''
    args.prefix += 'lr_exp{}_'.format(args.lr_exp) if not args.lr_exp == 1.0 else ""
    if args.optimizer == "sgd":
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, 
                        args.prefix + args.optimizer + str(args.lr)+'g'+str(args.gamma)+'momentum'+str(args.momentum)+'wd'+str(args.wd),
                        settings.TIME_NOW)
    else:
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, 
                        args.prefix + args.optimizer + str(args.lr)+'g'+str(args.gamma)+'clip'+
                        str(args.clip)+'momentum'+str(args.momentum)+'beta'+str(args.beta)+'os'+
                        str(args.os)+'wd'+str(args.wd)+'wn'+str(args.wn)+'wb'+str(int(args.wb))+'nlr'+str(int(args.nlr)),
                        settings.TIME_NOW)

    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)
    if args.optimizer == "sgd":
        writer = SummaryWriter(log_dir=os.path.join(
                            settings.LOG_DIR, args.net,
                            args.prefix + args.optimizer + str(args.lr)+'g'+str(args.gamma)+'momentum'+str(args.momentum)+'wd'+str(args.wd),
                            settings.TIME_NOW))
    else:
        writer = SummaryWriter(log_dir=os.path.join(
                            settings.LOG_DIR, args.net,
                            args.prefix + args.optimizer + str(args.lr)+'g'+str(args.gamma)+'clip'+
                            str(args.clip)+'momentum'+str(args.momentum)+'beta'+str(args.beta)+'os'+
                            str(args.os)+'wd'+str(args.wd)+'wn'+str(args.wn)+'wb'+str(int(args.wb))+'nlr'+str(int(args.nlr)),
                            settings.TIME_NOW))
    input_tensor = torch.Tensor(1, 3, 32, 32).cuda()
    writer.add_graph(net, input_tensor)

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    for epoch in range(1, settings.EPOCH):
        if epoch > args.warm:
            train_scheduler.step(epoch)
        writer.add_scalar("lr",optimizer.param_groups[0]['lr'],epoch)
        if args.loss == "ce" and args.grad:
            n = 2
            print("using weights: {}".format(weights[min(99,(epoch-1)*n)]))
            loss_function = nn.CrossEntropyLoss(weight=weights[min(99,(epoch-1)*n)])
        train(epoch)
        #optimizer.clean_weight()
        acc = eval_training(dataloader=cifar100_test_loader,train=False,epoch=epoch)
        acc_train = eval_training(dataloader=cifar100_training_loader,train=True,epoch=epoch)
        print(writer.log_dir)
        #start to save best performance model after learning rate decay to 0.01
        if epoch > settings.MILESTONES[1] and best_acc < acc:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='best'))
            best_acc = acc
            continue

        if not epoch % settings.SAVE_EPOCH:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='regular'))

    writer.close()
