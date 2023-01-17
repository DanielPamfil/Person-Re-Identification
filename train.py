
import argparse
import os
import time
import sys

from torchreid.utils import Logger
from torchreid.data.transforms import get_costum_transformer


parser = argparse.ArgumentParser(description='')

parser.add_argument('--path', default='')
parser.add_argument('--gpu', default=None, type=int, help='ID of the GPU ')

parser.add_argument('--snapshots_dir', type=str, default='snapshots')
parser.add_argument('logs_dir', type=str, default='logs')

def main():
    args = parser.parse_args()

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