"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import random
import math
import time
import copy
import warnings
import sys
import argparse
import shutil
import os.path as osp

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torch.nn.functional as F

sys.path.append('../../..')
from dalib.adaptation.daemda import FaultClassifier, WeightedDomainAdversarialLoss
from common.utils.data import ForeverDataIterator
from common.utils.metric import accuracy
from common.utils.meter import AverageMeter, ProgressMeter
from common.utils.logger import CompleteLogger
from common.utils.analysis import collect_feature, tsne, a_distance

sys.path.append('.')
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args: argparse.Namespace):
    logger = CompleteLogger(args.log, args.phase)
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cudnn.benchmark = True

    # Data loading code
    train_source_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, args.class_names = \
        utils.get_dataset(args.data, args.root, args.source, args.target)
    train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)

    # create model
    print("=> using model '{}'".format(args.arch))
    backbone = utils.get_model(args.arch, pretrain=not args.scratch)
    len_backbone = len(list(backbone.parameters()))
    pool_layer = None
    classifier = FaultClassifier(backbone, num_classes, bottleneck_dim=args.bottleneck_dim, finetune=not args.scratch,
                                 pool_layer=pool_layer).to(device)
    all_parameters = classifier.get_parameters()
    # define optimizer and lr scheduler
    # optimizer = SGD(all_parameters, args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    optimizer = Adam(all_parameters, args.lr, weight_decay=args.weight_decay)
    lr_scheduler = LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))

    weighted_adv_loss = WeightedDomainAdversarialLoss().to(device)
    # resume from the best checkpoint
    if args.phase != 'train':
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        classifier.load_state_dict(checkpoint)

    # analysis the model
    if args.phase == 'analysis':
        # extract features from both domains
        feature_extractor = nn.Sequential(classifier.backbone, classifier.pool_layer, classifier.bottleneck).to(device)
        source_feature = collect_feature(train_source_loader, feature_extractor, device)
        target_feature = collect_feature(train_target_loader, feature_extractor, device)
        # plot t-SNE
        tSNE_filename = osp.join(logger.visualize_directory, 'TSNE.svg')
        tsne.visualize(source_feature, target_feature, tSNE_filename)
        print("Saving t-SNE to", tSNE_filename)
        # calculate A-distance, which is a measure for distribution discrepancy
        A_distance = a_distance.calculate(source_feature, target_feature, device)
        print("A-distance =", A_distance)
        return

    if args.phase == 'test':
        acc1 = utils.validate(test_loader, classifier, args, device)
        print(acc1)
        return

    # get label distribution
    s_label_distribution = utils.get_label_distribution(train_source_loader, num_classes).to(device)
    t_output_distribution = copy.deepcopy(s_label_distribution)
    # start training
    best_acc1 = 0.
    for epoch in range(args.epochs):
        print('s_label_distribution', s_label_distribution)
        print('t_output_distribution', t_output_distribution)
        # print("lr:", lr_scheduler.get_last_lr()[0])
        # train for one epoch
        train(train_source_iter, train_target_iter, classifier, weighted_adv_loss, s_label_distribution,
              t_output_distribution, len_backbone, optimizer, lr_scheduler, num_classes, epoch, args)

        # evaluate on validation set
        acc1 = utils.validate(val_loader, classifier, args, device)

        # remember best acc@1 and save checkpoint
        torch.save(classifier.state_dict(), logger.get_checkpoint_path('latest'))
        if acc1 > best_acc1:
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
        best_acc1 = max(acc1, best_acc1)
        # update label distribution of target domain
        t_output_distribution = utils.get_output_distribution(classifier, test_loader, num_classes, device)

    print("best_acc1 = {:3.1f}".format(best_acc1))

    # evaluate on test set
    classifier.load_state_dict(torch.load(logger.get_checkpoint_path('best')))
    acc1 = utils.validate(test_loader, classifier, args, device)
    print("test_acc1 = {:3.1f}".format(acc1))

    logger.close()


def train(train_source_iter: ForeverDataIterator, train_target_iter: ForeverDataIterator, model: FaultClassifier,
          weighted_adv_loss: nn.Module, s_label_distribution, t_output_distribution, len_backbone: int, optimizer: SGD,
          lr_scheduler: LambdaLR, num_classes: int, epoch: int, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':3.1f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    trans_losses = AverageMeter('Trans Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')
    domain_accs = AverageMeter('Domain Acc', ':3.1f')
    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, trans_losses, cls_accs, domain_accs],
        prefix="Epoch: [{}]".format(epoch))

    # trade_off between target_entropy_loss and cls_loss
    lam = 2 / (1 + math.exp(-1. * 10. * epoch / args.epochs)) - 1

    # switch to train mode
    model.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        x_s, labels_s = next(train_source_iter)
        x_t, _ = next(train_target_iter)

        x_s = x_s.to(device)
        x_t = x_t.to(device)
        labels_s = labels_s.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # # compute output
        _, (y_s, d_s) = model(x_s)
        _, (y_t, d_t) = model(x_t)

        cls_loss = F.cross_entropy(y_s, labels_s)

        y_t = torch.softmax(y_t, dim=1)
        weight_adv = utils.get_adv_weight(s_label_distribution, t_output_distribution, labels_s, y_t)
        transfer_loss = weighted_adv_loss(d_s, d_t, weight_adv)
        target_entropy_loss = -torch.mean((y_t * torch.log(y_t + 1e-6)).sum(dim=1))

        loss = cls_loss + lam * target_entropy_loss * args.trade_off

        cls_acc = accuracy(y_s, labels_s)[0]
        domain_acc = weighted_adv_loss.domain_discriminator_accuracy

        losses.update(loss.item(), x_s.size(0))
        cls_accs.update(cls_acc, x_s.size(0))
        domain_accs.update(domain_acc, x_s.size(0))
        trans_losses.update(transfer_loss.item(), x_s.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        # 1.
        (transfer_loss * 0.).backward(retain_graph=True)
        adv_grad = []
        for param in model.parameters():
            if param.grad is not None:
                adv_grad.append(param.grad.data.clone())
            else:
                adv_grad.append(None)

        optimizer.zero_grad()
        loss.backward()
        # 4 update parameters
        for index, param in enumerate(model.parameters()):
            if param.grad is not None:
                if index < len_backbone:
                    param.grad.data = param.grad.data - adv_grad[index]
                else:
                    param.grad.data = param.grad.data + adv_grad[index]
        optimizer.step()
        lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='daemda for Unsupervised Domain Adaptation')
    # dataset parameters
    parser.add_argument('root', metavar='DIR', help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', choices=utils.get_dataset_names(),
                        help='dataset: ' + ' | '.join(utils.get_dataset_names()) + ' (default: Office31)')
    parser.add_argument('-s', '--source', help='source domain(s)', nargs='+')
    parser.add_argument('-t', '--target', help='target domain(s)', nargs='+')
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', choices=utils.get_model_names(),
                        help='backbone architecture: ' + ' | '.join(utils.get_model_names()) +
                             ' (default: resnet18)')
    parser.add_argument('--bottleneck-dim', default=100, type=int, help='Dimension of bottleneck')
    parser.add_argument('--no-pool', action='store_true', help='no pool layer after the feature extractor.')
    parser.add_argument('--scratch', action='store_true',
                        help='whether train from scratch.')  # action='store_true' 效果等于default=False
    # training parameters
    parser.add_argument('-lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-gamma', default=0.001, type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
    parser.add_argument('--trade-off', default=1., type=float, help='the trade-off hyper-parameter for transfer loss')
    parser.add_argument('-b', '--batch-size', default=220, type=int, metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-3)',
                        dest='weight_decay')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=1000, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--per-class-eval', action='store_true',
                        help='whether output per-class accuracy during evaluation')
    parser.add_argument("--log", type=str, default='cdan',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    args = parser.parse_args()
    main(args)
