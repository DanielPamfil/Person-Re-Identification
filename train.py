
import argparse
import os
import time
import sys

import torch
import torch.cuda
import torchvision.models as models
import torch.nn as nn

from torchreid.utils import Logger
from torchreid.data.transforms import get_costum_transformer, get_validation_transformer
from torchreid.data.datasets.dataset import CostumDataset, StandardDataset
from torchreid.models.MoCo import MoCo
from torchreid.data.sampler import InferenceSampler
from torchreid.utils.avgmeter import AverageMeter, ProgressMeter
from torchreid.utils.torchtools import adjust_learning_rate, save_checkpoint_light
from torch.cuda.amp import GradScaler
from torchreid.metrics.accuracy import accuracy


parser = argparse.ArgumentParser(description='')

parser.add_argument('--dataset_path', type=str, default='')
parser.add_argument('--evaluation_path', type=str, default='')

parser.add_argument('--key_path', type=str, default='')

parser.add_argument('--gpu', default=None, type=int, help='ID of the GPU ')

parser.add_argument('--snapshots_dir', type=str, default='snapshots')
parser.add_argument('--logs_dir', type=str, default='logs')

parser.add_argument('--auto_resume', type=bool, default=True)

parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--epochs', default=100, type=int)

def main():
    args = parser.parse_args()

    # Hyperparameters
    lr = 0.3
    momentum = 0.9
    weight_decay = 1e-4

    # Momentum Contrast Hyperparameters
    moco_dimension = 128
    moco_queue = 65536
    moco_momentum = 0.999
    moco_temperature = 0.1

    # Confirm the used gpu
    if args.gpu is not None:
        print('You are using gpu number: ', args.gpu)

    if not os.path.exists(args.snapshots_dir):
        os.makedirs(args.snapshots_dir)
    if not os.path.exists(args.logs_dir):
        os.makedirs(args.logs_dir)
    log_text_name = "train_log" + time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime()) + '.txt'
    sys.stdout = Logger(os.path.join(args.logs_dir, log_text_name))

    # Data loading
    train_transformer = get_costum_transformer()
    val_transformer = get_validation_transformer()

    # Load datasets
    train_dataset = CostumDataset(args.dataset_path, args.key_path, train_transformer)
    val_dataset = StandardDataset(args.evaluation_path, dataset_name='market', mode='test', transform=val_transformer)

    # Create the model
    model = MoCo(models.resnet50(), dim=moco_dimension, K=moco_queue, m=moco_momentum, T=moco_temperature, mlp=True)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)

        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], broadcast_buffers=False)

    else:
        model.cuda()
        model = nn.parallel.DistributedDataParallel(model)

    # Loss criterion
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # Define Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr, momentum=momentum, weight_decay=weight_decay)

    if args.auto_resume:
        latest_checkpoint = os.path.join(args.snap_dir, 'ckpt_latest.pth')
        if os.path.isfile(latest_checkpoint):
            args.resume = latest_checkpoint
        else:
            snapshots = os.listdir(args.snapshots_dir)
            snapshots = [x for x in snapshots if x.endswith('.pth')]
            if len(snapshots) > 0:
                max_snap_epoch = 0
                for snap in snapshots:
                    tmp_epoch = int(snap[:-4].split('_')[1])
                    if tmp_epoch > max_snap_epoch:
                        max_snap_epoch = tmp_epoch
                latest_checkpoint = 'ckpt_{:04d}.pth'.format(max_snap_epoch)
                args.resume = os.path.join(args.snap_dir, latest_checkpoint)
            else:
                print('Does not exist any previus snapshot in the following directory')

    if args.resume:
        if os.path.isfile(args.resume):
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=>rank[{}] loaded checkpoint '{}' (epoch {})"
                  .format(args.rank, args.resume, checkpoint['epoch']))
        else:
            print('Checkpoint not found')

    # Initialize the samplers
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = InferenceSampler(len(val_dataset))

    # Initialize the data loaders
    train_loader = torch.utils.data.Dataloader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True )
    validation_loader = torch.utils.data.Dataloader(val_dataset, batch_size=128, num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=False)

    for epoch in range(start_epoch, args.epochs):
        print('Epoch: ', epoch, 'of', args.epochs )

        train_sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, epoch, args)

        start_of_epoch = time.time()

        loss, accuracy = train(train_loader, model, criterion, optimizer, epoch, args)

        print('Epoch = ', epoch, 'Loss = ', loss, 'Accuracy =', accuracy)

        latest = os.path.join(args.snapshots_dir, 'ckpt_latest.pth')
        filename = os.path.join(args.snapshots_dir, 'ckpt_{:04d}.pth'.format(epoch))
        print('Save checkpoint')
        save_checkpoint_light()













def train(train_loader, model, criterion, optimizer, epoch, args):
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Acc_1', ':.2f')
    top5 = AverageMeter('Acc_5', ':.2f')
    batch_time = AverageMeter('T', ':.3f')
    data_time = AverageMeter('DT', ':.3f')

    arr = [batch_time, data_time, losses, top1, top5]
    progress = ProgressMeter(len(train_loader), arr, prefix="Epoch: [{}]".format(epoch))

    if args.print_freq < 1:
        print_freq = max(-args.print_freq, 1)

    else:
        print_freq = max((len(train_loader) + args.print_freq - 1), 1)

    # switch to train mode
    model.train()
    end = time.time()

    if args.mix:
        scaler = GradScaler()
    else:
        None

    for i, (images, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)

        # compute output
        if args.mix:
            output, target = model(im_q=images[0], im_k=images[1])
            loss = criterion(model(im_q=images[0], im_k=images[1]))
        else:
            output, target = model(im_q=images[0], im_k=images[1])
            loss = criterion(output, target)

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        accuracy1, accurracy5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images[0].size(0))
        top1.update(accuracy1[0], images[0].size(0))
        top5.update(accurracy5[0], images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        if args.mix:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # measure elapsed time
        end = time.time()
        batch_time.update(time.time() - end)

        if i % print_freq == 0 and args.rank == 0:
            progress.display(i)

    return losses.avg, top1.avg