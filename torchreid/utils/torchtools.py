from __future__ import division, print_function, absolute_import
import pickle
import shutil
import os.path as osp
import warnings
from functools import partial
from collections import OrderedDict
import torch
import torch.nn as nn
import math
import torch.distributed as dist

from .tools import mkdir_if_missing

__all__ = [
    'save_checkpoint', 'load_checkpoint', 'resume_from_checkpoint',
    'open_all_layers', 'open_specified_layers', 'count_num_param',
    'load_pretrained_weights'
]


def save_checkpoint(
    state, save_dir, is_best=False, remove_module_from_keys=False
):
    r"""Saves checkpoint.

    Args:
        state (dict): dictionary.
        save_dir (str): directory to save checkpoint.
        is_best (bool, optional): if True, this checkpoint will be copied and named
            ``model-best.pth.tar``. Default is False.
        remove_module_from_keys (bool, optional): whether to remove "module."
            from layer names. Default is False.

    Examples::
        >>> state = {
        >>>     'state_dict': model.state_dict(),
        >>>     'epoch': 10,
        >>>     'rank1': 0.5,
        >>>     'optimizer': optimizer.state_dict()
        >>> }
        >>> save_checkpoint(state, 'log/my_model')
    """
    mkdir_if_missing(save_dir)
    if remove_module_from_keys:
        # remove 'module.' in state_dict's keys
        state_dict = state['state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]
            new_state_dict[k] = v
        state['state_dict'] = new_state_dict
    # save
    epoch = state['epoch']
    fpath = osp.join(save_dir, 'model.pth.tar-' + str(epoch))
    torch.save(state, fpath)
    print('Checkpoint saved to "{}"'.format(fpath))
    if is_best:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), 'model-best.pth.tar'))

def save_checkpoint_light(state, save_cur, cur_name='ckpt.pth', lastest_name='latest.pth'):
    torch.save(state, lastest_name)
    if save_cur:
        shutil.copyfile(lastest_name, cur_name)
def load_checkpoint(fpath):
    r"""Loads checkpoint.

    ``UnicodeDecodeError`` can be well handled, which means
    python2-saved files can be read from python3.

    Args:
        fpath (str): path to checkpoint.

    Returns:
        dict

    Examples::
        >>> from torchreid.utils import load_checkpoint
        >>> fpath = 'log/my_model/model.pth.tar-10'
        >>> checkpoint = load_checkpoint(fpath)
    """
    if fpath is None:
        raise ValueError('File path is None')
    fpath = osp.abspath(osp.expanduser(fpath))
    if not osp.exists(fpath):
        raise FileNotFoundError('File is not found at "{}"'.format(fpath))
    map_location = None if torch.cuda.is_available() else 'cpu'
    try:
        checkpoint = torch.load(fpath, map_location=map_location)
    except UnicodeDecodeError:
        pickle.load = partial(pickle.load, encoding="latin1")
        pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
        checkpoint = torch.load(
            fpath, pickle_module=pickle, map_location=map_location
        )
    except Exception:
        print('Unable to load checkpoint from "{}"'.format(fpath))
        raise
    return checkpoint


def resume_from_checkpoint(fpath, model, optimizer=None, scheduler=None):
    r"""Resumes training from a checkpoint.

    This will load (1) model weights and (2) ``state_dict``
    of optimizer if ``optimizer`` is not None.

    Args:
        fpath (str): path to checkpoint.
        model (nn.Module): model.
        optimizer (Optimizer, optional): an Optimizer.
        scheduler (LRScheduler, optional): an LRScheduler.

    Returns:
        int: start_epoch.

    Examples::
        >>> from torchreid.utils import resume_from_checkpoint
        >>> fpath = 'log/my_model/model.pth.tar-10'
        >>> start_epoch = resume_from_checkpoint(
        >>>     fpath, model, optimizer, scheduler
        >>> )
    """
    print('Loading checkpoint from "{}"'.format(fpath))
    checkpoint = load_checkpoint(fpath)
    model.load_state_dict(checkpoint['state_dict'])
    print('Loaded model weights')
    if optimizer is not None and 'optimizer' in checkpoint.keys():
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('Loaded optimizer')
    if scheduler is not None and 'scheduler' in checkpoint.keys():
        scheduler.load_state_dict(checkpoint['scheduler'])
        print('Loaded scheduler')
    start_epoch = checkpoint['epoch']
    print('Last epoch = {}'.format(start_epoch))
    if 'rank1' in checkpoint.keys():
        print('Last rank1 = {:.1%}'.format(checkpoint['rank1']))
    return start_epoch


def adjust_learning_rate(optimizer, epoch, lr_def=0.3, warmup_epochs=0, cos=1, tot_epochs=200, schedule='60,80'):
    """Decay the learning rate based on schedule"""
    #lr = args.lr * args.lr_mult
    lr = lr_def
    if epoch < warmup_epochs:
        # warm up
        lr = lr_def + (lr_def * 1 - lr_def) / warmup_epochs * epoch
    elif cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / tot_epochs))
    else:  # stepwise lr schedule
        for milestone in schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def set_bn_to_eval(m):
    r"""Sets BatchNorm layers to eval mode."""
    # 1. no update for running mean and var
    # 2. scale and shift parameters are still trainable
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def open_all_layers(model):
    r"""Opens all layers in model for training.

    Examples::
        >>> from torchreid.utils import open_all_layers
        >>> open_all_layers(model)
    """
    model.train()
    for p in model.parameters():
        p.requires_grad = True


def open_specified_layers(model, open_layers):
    r"""Opens specified layers in model for training while keeping
    other layers frozen.

    Args:
        model (nn.Module): neural net model.
        open_layers (str or list): layers open for training.

    Examples::
        >>> from torchreid.utils import open_specified_layers
        >>> # Only model.classifier will be updated.
        >>> open_layers = 'classifier'
        >>> open_specified_layers(model, open_layers)
        >>> # Only model.fc and model.classifier will be updated.
        >>> open_layers = ['fc', 'classifier']
        >>> open_specified_layers(model, open_layers)
    """
    if isinstance(model, nn.DataParallel):
        model = model.module

    if isinstance(open_layers, str):
        open_layers = [open_layers]

    for layer in open_layers:
        assert hasattr(
            model, layer
        ), '"{}" is not an attribute of the model, please provide the correct name'.format(
            layer
        )

    for name, module in model.named_children():
        if name in open_layers:
            module.train()
            for p in module.parameters():
                p.requires_grad = True
        else:
            module.eval()
            for p in module.parameters():
                p.requires_grad = False


def count_num_param(model):
    r"""Counts number of parameters in a model while ignoring ``self.classifier``.

    Args:
        model (nn.Module): network model.

    Examples::
        >>> from torchreid.utils import count_num_param
        >>> model_size = count_num_param(model)

    .. warning::
        
        This method is deprecated in favor of
        ``torchreid.utils.compute_model_complexity``.
    """
    warnings.warn(
        'This method is deprecated and will be removed in the future.'
    )

    num_param = sum(p.numel() for p in model.parameters())

    if isinstance(model, nn.DataParallel):
        model = model.module

    if hasattr(model,
               'classifier') and isinstance(model.classifier, nn.Module):
        # we ignore the classifier because it is unused at test time
        num_param -= sum(p.numel() for p in model.classifier.parameters())

    return num_param


def load_pretrained_weights(model, weight_path):
    r"""Loads pretrianed weights to model.

    Features::
        - Incompatible layers (unmatched in name or size) will be ignored.
        - Can automatically deal with keys containing "module.".

    Args:
        model (nn.Module): network model.
        weight_path (str): path to pretrained weights.

    Examples::
        >>> from torchreid.utils import load_pretrained_weights
        >>> weight_path = 'log/my_model/model-best.pth.tar'
        >>> load_pretrained_weights(model, weight_path)
    """
    checkpoint = load_checkpoint(weight_path)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []

    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:] # discard module.

        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)

    if len(matched_layers) == 0:
        warnings.warn(
            'The pretrained weights "{}" cannot be loaded, '
            'please check the key names manually '
            '(** ignored and continue **)'.format(weight_path)
        )
    else:
        print(
            'Successfully loaded pretrained weights from "{}"'.
            format(weight_path)
        )
        if len(discarded_layers) > 0:
            print(
                '** The following layers are discarded '
                'due to unmatched keys or layer size: {}'.
                format(discarded_layers)
            )

@torch.no_grad()
def pad_to_max_size(tensor):
    local_size = torch.tensor([tensor.shape[0]], dtype=torch.int64, device=tensor.device)
    size_list = [torch.zeros([1], dtype=torch.int64, device=tensor.device)
                 for _ in range(dist.get_world_size())]
    dist.all_gather(size_list, local_size, async_op=False)
    max_size = torch.cat(size_list, dim=0).max().item()
    if local_size < max_size:
        padding = torch.zeros((max_size - local_size, *tensor.shape[1:]),
                              dtype=tensor.dtype, device=tensor.device)
        flag = torch.cat([torch.ones(local_size, dtype=torch.uint8, device=tensor.device),
                          torch.zeros(max_size - local_size, dtype=torch.uint8, device=tensor.device)], dim=0)
        tensor = torch.cat((tensor, padding), dim=0)
    else:
        flag = torch.ones(local_size, dtype=torch.uint8, device=tensor.device)
    return tensor, flag

@torch.no_grad()
def gather_tensors(tensor):
    """ Gather tensors from different workers with different shape
    tensors on different devices must have the same data dims except the bacth_num
    """
    tensor, flag = pad_to_max_size(tensor)
    tensors_gather = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    flags_gather = [torch.zeros_like(flag) for _ in range(dist.get_world_size())]
    dist.all_gather(tensors_gather, tensor, async_op=False)
    dist.all_gather(flags_gather, flag, async_op=False)

    outputs = torch.cat(tensors_gather, dim=0)
    flags = torch.cat(flags_gather, dim=0)
    output = outputs[flags > 0]
    return output