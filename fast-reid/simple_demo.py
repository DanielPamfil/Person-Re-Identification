# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import argparse
import logging
import sys
import os

import cv2
import pickle
import numpy as np
import torch
import tqdm
from torch.backends import cudnn

sys.path.append('.')

from fastreid.evaluation import evaluate_rank
from fastreid.config import get_cfg
from fastreid.utils.logger import setup_logger
from fastreid.data import build_reid_test_loader
from demo.predictor import FeatureExtractionDemo
from fastreid.utils.visualizer import Visualizer

# import some modules added in project
# for example, add partial reid like this below
# sys.path.append("projects/PartialReID")
# from partialreid import *

cudnn.benchmark = True
setup_logger(name="fastreid")

logger = logging.getLogger('fastreid.visualize_result')


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # add_partialreid_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Feature extraction with reid models")
    parser.add_argument(
        "--config-file",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='if use multiprocess for feature extraction.'
    )
    parser.add_argument(
        "--dataset-name",
        help="a test dataset name for visualizing ranking list."
    )
    parser.add_argument(
        "--output",
        default="./vis_rank_list",
        help="a file or directory to save rankling list result.",

    )
    parser.add_argument(
        "--vis-label",
        action='store_true',
        help="if visualize label of query instance"
    )
    parser.add_argument(
        "--num-vis",
        default=100,
        help="number of query images to be visualized",
    )
    parser.add_argument(
        "--rank-sort",
        default="ascending",
        help="rank order of visualization images by AP metric",
    )
    parser.add_argument(
        "--label-sort",
        default="ascending",
        help="label order of visualization images by cosine similarity metric",
    )
    parser.add_argument(
        "--max-rank",
        default=10,
        help="maximum number of rank list to be visualized",
    )
    parser.add_argument(
        "--query",
        default="C:/Users/danie/PycharmProjects/ComputerVision/LUPerson/fast-reid/media/video3/24.0.jpg",
        help="a file or directory to save rankling list result.",

    )
    parser.add_argument(
        "--gallery",
        default="C:/Users/danie/PycharmProjects/ComputerVision/LUPerson/fast-reid/media/video3/frame-162.jpg",
        help="a file or directory to save rankling list result.",

    )
    parser.add_argument(
        "--video-gallery",
        default="C:/Users/danie/PycharmProjects/ComputerVision/LUPerson/fast-reid/media/video3",
        help="a file or directory to save rankling list result.",

    )

    parser.add_argument(
        "--video-query",
        default="C:/Users/danie/PycharmProjects/ComputerVision/LUPerson/fast-reid/media/video3",
        help="a file or directory to save rankling list result.",

    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    cfg = setup_cfg(args)
    demo = FeatureExtractionDemo(cfg, parallel=args.parallel)

    logger.info("Start extracting image features")
    """
    q = cv2.imread(args.query)
    q = cv2.cvtColor(q, cv2.COLOR_BGR2RGB)
    g = cv2.imread(args.gallery)
    g = cv2.cvtColor(g, cv2.COLOR_BGR2RGB)

    feats_query = demo.run_on_image(q)
    feats_gallery = demo.run_on_image(g)
    """

    print()
    count_query = 0
    dictImage = {}
    for video_q in os.listdir(args.video_query):
        video_q_path = os.path.join(args.video_query, video_q)
        print(video_q_path)
        vq = cv2.imread(video_q_path)
        vq = cv2.cvtColor(vq, cv2.COLOR_BGR2RGB)
        feats_video_q = demo.run_on_image(vq)
        for person in os.listdir(args.video_gallery):
            count_gallery = 0
            person_path = os.path.join(args.video_gallery, person)
            for video_g in os.listdir(person_path):
                video_g_path = os.path.join(person_path, video_g)
                vg = cv2.imread(video_g_path)
                vg = cv2.cvtColor(vg, cv2.COLOR_BGR2RGB)
                feats_video_g = demo.run_on_image(vg)
                distance = 1 - torch.mm(feats_video_q, feats_video_g.t())
                #dictImage[person+"/"+video_g] = distance.item()
                print('video', video_g_path)
                #print('distance', distance)
                if distance[0] < 0.5:
                    dictImage[person + "/" + video_g] = distance.item()
                    #print('The person is the same')
                    count_gallery += 1
                #else:
                    #print('The person is not the same')
                if count_gallery >= 5:
                    print('The person is in the video')
                    count_query += 1
                    break
        if count_query >= 5:
            dictImage = sorted(dictImage.items(), key = lambda x: x[1])
            #args.video_gallery[:args.video_gallery.rindex("/")
            #with open(args.video_gallery+'/'+'dictImage.pkl', 'wb') as f:
            with open(args.video_gallery[:args.video_gallery.rindex("/")] + '/' + 'dictImage.pkl', 'wb') as f:
                pickle.dump(dictImage, f)
            print('The person is in the video 5 times')
            sys.exit(0)
            break
    sys.exit(1)


