import argparse
import torch
import torch.distributed as dist
from torchreid.metrics.score import reid_score
from torchreid.utils.torchtools import gather_tensors

def evaluate(val_loader, model, args):
    # Setting the model in evaluation mode
    model.eval()
    # Initializing the lists to store the features, pids, cameras and flags of the validation data
    pids, features, flags, cameras, flags = [], [], [], []

    # Running the model in evaluation mode, with no gradients
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            # Loading the validation data
            img, pid, cam, flag = data[0:4]
            # Calculating the features using the model
            feat = model(img.cuda(args.gpu, non_blocking=True))
            # Appending the features, pids, cameras and flags to their respective lists
            features.append(feat)
            pids.append(pid.cuda(args.gpu, non_blocking=True))
            cameras.append(cam.cuda(args.gpu, non_blocking=True))
            flags.append(flag.cuda(args.gpu, non_blocking=True))
    # Concatenating the features, pids, cameras and flags
    pids  = torch.cat(pids, dim=0)
    cameras  = torch.cat(cameras, dim=0)
    features = torch.cat(features, dim=0)
    flags = torch.cat(flags, dim=0)

    # Synchronizing the process
    dist.barrier()

    # Gathering the tensors of features, pids and cameras
    feats = gather_tensors(features)
    pids  = gather_tensors(pids)
    cams  = gather_tensors(cameras)
    flags = gather_tensors(flags)
    # Only the process with rank 0 will perform the evaluation
    if args.rank == 0:
        # Indexing the query data
        query_idx = flags > 0
        # Indexing the gallery data
        gall_idx = flags < 1
        # Calculating the mAP and cmc scores using the reid_score function
        mAP, cmc = reid_score(features[query_idx], pids[query_idx], cameras[query_idx],
                                      features[gall_idx], pids[gall_idx], cameras[gall_idx])
        return mAP, cmc
    else:
        return 0, 0

