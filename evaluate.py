import argparse
import torch

def evaluate(val_loader, model, args):
    model.eval()
    feats, pids, cams, flags = [], [], [], []
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            img, pid, cam, flag = data[0:4]
            feat = model(img.cuda(args.gpu, non_blocking=True))
            feats.append(feat)
            pids.append(pid.cuda(args.gpu, non_blocking=True))
            cams.append(cam.cuda(args.gpu, non_blocking=True))
            flags.append(flag.cuda(args.gpu, non_blocking=True))
    feats = torch.cat(feats, dim=0)
    pids  = torch.cat(pids, dim=0)
    cams  = torch.cat(cams, dim=0)
    flags = torch.cat(flags, dim=0)
    dist.barrier()
    #######################################
    # For multiprocessing, don't use for now
    #feats = gather_tensors(feats)
    #pids  = gather_tensors(pids)
    #cams  = gather_tensors(cams)
    #flags = gather_tensors(flags)
    ########################################
    if args.rank == 0:
        q_idx = flags > 0
        g_idx = flags < 1
        mAP, cmc = compute_reid_score(feats[q_idx], pids[q_idx], cams[q_idx],
                                      feats[g_idx], pids[g_idx], cams[g_idx])
        return mAP, cmc
    else:
        return 0, 0

