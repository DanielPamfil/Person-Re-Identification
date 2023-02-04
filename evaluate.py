import argparse
import torch
import torch.distributed as dist
from torchreid.metrics.score import reid_score
from torchreid.utils.torchtools import gather_tensors

def evaluate(val_loader, model, args):
    model.eval()
    pids, features, flags, cameras, flags = [], [], [], []
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            img, pid, cam, flag = data[0:4]
            feat = model(img.cuda(args.gpu, non_blocking=True))
            features.append(feat)
            pids.append(pid.cuda(args.gpu, non_blocking=True))
            cameras.append(cam.cuda(args.gpu, non_blocking=True))
            flags.append(flag.cuda(args.gpu, non_blocking=True))
    pids  = torch.cat(pids, dim=0)
    cameras  = torch.cat(cameras, dim=0)
    features = torch.cat(features, dim=0)
    ######da rivedere
    flags = torch.cat(flags, dim=0)
    dist.barrier()
    feats = gather_tensors(features)
    pids  = gather_tensors(pids)
    cams  = gather_tensors(cameras)
    flags = gather_tensors(flags)
    if args.rank == 0:
        query_idx = flags > 0
        gall_idx = flags < 1
        mAP, cmc = reid_score(features[query_idx], pids[query_idx], cameras[query_idx],
                                      features[gall_idx], pids[gall_idx], cameras[gall_idx])
        return mAP, cmc
    else:
        return 0, 0

