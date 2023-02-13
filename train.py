import argparse
import os
import time
import sys

import torch
import torch.cuda
import torchvision.models as models
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.utils.data

from torchreid.utils import Logger
from torchreid.data.transforms import get_costum_transformer, get_validation_transformer
from torchreid.data.datasets.dataset import CostumDataset, StandardDataset
from torchreid.models.MoCo import MoCo
from torchreid.data.sampler import InferenceSampler
from torchreid.utils.avgmeter import AverageMeter, ProgressMeter
from torchreid.utils.torchtools import adjust_learning_rate, save_checkpoint_light
from torch.cuda.amp import GradScaler
from torchreid.metrics.accuracy import accuracy
from torchreid.utils.torchtools import gather_tensors

try:
    from torch.cuda.amp import autocast, GradScaler

    MIX_P = True
except Exception as exc:
    MIX_P = False


def main_function(gpu, ngpus_per_node, args):
    # Hyperparameters
    lr = 0.3
    momentum = 0.9
    weight_decay = 1e-4
    cos = 1
    warmup_epochs = 1

    # Training
    start_epoch = 0

    # distribuited parameters
    rank = 0
    dist_backend = 'gloo'
    dist_url = 'tcp://localhost:23456'
    world_size = 1
    workers = 10

    # Momentum Contrast Hyperparameters
    moco_dimension = 128
    moco_queue = 65536
    moco_momentum = 0.999
    moco_temperature = 0.1

    # Confirm GPU being used
    if args.gpu is not None:
        print('You are using gpu number: ', args.gpu)

    # Create directories for snapshots and logs if they do not exist
    if not os.path.exists(args.snapshots_dir):
        os.makedirs(args.snapshots_dir)
    if not os.path.exists(args.logs_dir):
        os.makedirs(args.logs_dir)

    # Name the log file with a timestamp
    log_text_name = "train_log" + time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime()) + '.txt'
    # Redirect the standard output to the log file
    sys.stdout = Logger(os.path.join(args.logs_dir, log_text_name))

    # Initialize process group for distributed training
    dist.init_process_group(backend=dist_backend, init_method=dist_url, world_size=world_size, rank=rank)

    # Load data and apply transformations
    train_transformer = get_costum_transformer()
    val_transformer = get_validation_transformer()

    # Load datasets
    # Create a custom dataset with the provided arguments
    train_dataset = CostumDataset(args.dataset_path, args.key_path, train_transformer)
    # Create a standard dataset for validation purposes
    val_dataset = StandardDataset(args.evaluation_path, dataset_name='market1501', mode='test',
                                  transform=val_transformer)

    # Print the length of the train dataset
    print("Len dataset: ", len(train_dataset))
    # Create a MoCo model using ResNet50 as the base model with the provided arguments
    model = MoCo(models.resnet50, dim=moco_dimension, K=moco_queue, m=moco_momentum, T=moco_temperature, mlp=True)

    # Check if the GPU device is specified
    if args.gpu is not None:
        # Set the specified GPU device
        torch.cuda.set_device(args.gpu)
        # Move the model to the GPU device
        model.cuda(args.gpu)
        # Wrap the model with nn.parallel.DistributedDataParallel to support multi-GPU training
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], broadcast_buffers=False)
    else:
        # Move the model to the GPU device if GPU is not specified
        model.cuda()
        # Wrap the model with nn.parallel.DistributedDataParallel to support multi-GPU training
        model = nn.parallel.DistributedDataParallel(model)

    # Create a CrossEntropyLoss criterion and move it to the GPU device
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # Define the optimizer as Stochastic Gradient Descent with the provided learning rate, momentum, and weight decay
    optimizer = torch.optim.SGD(model.parameters(), lr, momentum=momentum, weight_decay=weight_decay)

    # Check if auto_resume is enabled
    if args.auto_resume:
        # Get the latest checkpoint file
        latest_checkpoint = os.path.join(args.snapshots_dir, 'ckpt_latest.pth')
        # Check if the latest checkpoint file exists
        if os.path.isfile(latest_checkpoint):
            # Set the resume argument to the latest checkpoint file
            args.resume = latest_checkpoint
        else:
            # Get the list of snapshots in the snapshots directory
            snapshots = os.listdir(args.snapshots_dir)
            # Filter the list to get only the .pth files
            snapshots = [x for x in snapshots if x.endswith('.pth')]
            # Check if there are snapshots
            if len(snapshots) > 0:
                # Initialize the max_snap_epoch variable
                max_snap_epoch = 0
                # Loop through the snapshots
                for snap in snapshots:
                    # Get the epoch of the snapshot
                    tmp_epoch = int(snap[:-4].split('_')[1])
                    # Check if the current snapshot's epoch is greater than the max_snap_epoch
                    if tmp_epoch > max_snap_epoch:
                        # Update the max_snap_epoch
                        max_snap_epoch = tmp_epoch
                latest_checkpoint = 'ckpt_{:04d}.pth'.format(max_snap_epoch)
                args.resume = os.path.join(args.snap_dir, latest_checkpoint)
            else:
                print('Does not exist any previus snapshot in the following directory')

    # Check if the resume argument is set
    if args.resume:
        # Check if the resume argument is a valid file
        if os.path.isfile(args.resume):
            # Check if the GPU is set
            if args.gpu is None:
                # Load the checkpoint without specifying the GPU
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            # Get the starting epoch from the checkpoint
            start_epoch = checkpoint['epoch']
            # Load the state dict of the model from the checkpoint
            model.load_state_dict(checkpoint['state_dict'])
            # Load the state dict of the optimizer from the checkpoint
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=>rank[{}] loaded checkpoint '{}' (epoch {})"
                  .format(args.rank, args.resume, checkpoint['epoch']))
        else:
            print('Checkpoint not found')

    # Initialize the distributed sampler to train the model on multiple GPUs
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    # Initialize the InferenceSampler to validate the model
    val_sampler = InferenceSampler(len(val_dataset))

    # Initialize the training data loader with the train_dataset, batch size of args.batch_size, shuffle the data if train_sampler is None, number of workers to load the data is workers, pin the memory for faster loading, use the train_sampler for sampling, drop the last batch if its size is smaller than batch_size
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=(train_sampler is None), num_workers=workers, pin_memory=True,
                                               sampler=train_sampler, drop_last=True)
    # Initialize the validation data loader with the val_dataset, batch size of 128, number of workers to load the data is workers, pin the memory for faster loading, use the val_sampler for sampling, do not drop the last batch if its size is smaller than batch_size
    validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, num_workers=workers, pin_memory=True,
                                                    sampler=val_sampler, drop_last=False)

    # Loop through all epochs
    for epoch in range(start_epoch, args.epochs):
        # Print the current epoch number and the total number of epochs
        print('Epoch: ', epoch, 'of', args.epochs)

        # Set the epoch number for the train_sampler
        train_sampler.set_epoch(epoch)

        # Adjust the learning rate for the optimizer
        adjust_learning_rate(optimizer, epoch)

        # Record the start time of the epoch
        start_of_epoch = time.time()

        # Train the model for one epoch and get the loss and accuracy values
        loss, accuracy = train(train_loader, model, criterion, optimizer, epoch, args)

        print('Epoch = ', epoch, 'Loss = ', loss, 'Accuracy =', accuracy)

        # Define the latest checkpoint file and the checkpoint file for the current epoch
        latest = os.path.join(args.snapshots_dir, 'ckpt_latest.pth')
        filename = os.path.join(args.snapshots_dir, 'ckpt_{:04d}.pth'.format(epoch))
        print('Save checkpoint')
        # Save the checkpoint for the current epoch only if the epoch number is divisible by 20 or it's the last epoch. Save the current checkpoint under filename, the latest checkpoint under latest.
        save_current = epoch % 20 == 0 or epoch == args.epochs - 1
        save_checkpoint_light({'epoch': epoch + 1, 'arch': 'resnet50', 'state_dict': model.state_dict(),
                               'optimizer': optimizer.state_dict()}, save_cur=save_current, cur_name=filename,
                              lastest_name=latest)


def train(train_loader, model, criterion, optimizer, epoch, args):
    # Define meters to keep track of average values of loss and accuracy during the training
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Acc_1', ':.2f')
    top5 = AverageMeter('Acc_5', ':.2f')
    batch_time = AverageMeter('T', ':.3f')
    data_time = AverageMeter('DT', ':.3f')

    # Store the meters in a list
    arr = [batch_time, data_time, losses, top1, top5]
    # Initialize the progress meter with the length of the train loader, the list of meters, and a prefix for each epoch
    progress = ProgressMeter(len(train_loader), arr, prefix="Epoch: [{}]".format(epoch))

    # Determine the frequency for printing the progress during the training
    if args.print_freq < 1:
        print_freq = max(-args.print_freq, 1)

    else:
        print_freq = max((len(train_loader) + args.print_freq - 1), 1)

    # Set the model to train mode
    model.train()
    # Store the time at the end of the data loading time
    end = time.time()

    # Initialize a GradScaler
    scaler = GradScaler()
    print(len(train_loader))

    # Loop through the training data in the train loader
    for i, (images, _) in enumerate(train_loader):
        # Measure the data loading time
        data_time.update(time.time() - end)

        # If GPU is used, move the images to GPU
        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)

        # Compute the output and target using the model
        with autocast():
            output, target = model(im_q=images[0], im_k=images[1])
            loss = criterion(output, target)

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # Compute the accuracy and record the loss
        accuracy1, accurracy5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images[0].size(0))
        top1.update(accuracy1[0], images[0].size(0))
        top5.update(accurracy5[0], images[0].size(0))

        # Zero out the gradient, compute the gradient of the loss, and do the SGD step
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Measure the elapsed time
        end = time.time()
        batch_time.update(time.time() - end)

    # Return the average loss and accuracy after all the iterations
    return losses.avg, top1.avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_path', type=str, default='')
    parser.add_argument('--evaluation_path', type=str, default='')

    parser.add_argument('--key_path', type=str, default='')

    parser.add_argument('--gpu', default=None, type=int, help='ID of the GPU ')

    parser.add_argument('--snapshots_dir', type=str, default='snapshots')
    parser.add_argument('--logs_dir', type=str, default='logs')

    parser.add_argument('--auto_resume', type=bool, default=True)
    parser.add_argument('--resume', default='', type=str)

    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=100, type=int)

    parser.add_argument('--print-freq', default=10, type=int)

    args = parser.parse_args()
    print("The script is running")
    mp.spawn(main_function, nprocs=1, args=(1, args))
