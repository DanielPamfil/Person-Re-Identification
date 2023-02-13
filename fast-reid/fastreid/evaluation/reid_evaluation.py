# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import copy
import logging
from collections import OrderedDict
from sklearn import metrics
import matplotlib.pyplot as plt
import os

import numpy as np
import torch
import torch.nn.functional as F

from .evaluator import DatasetEvaluator
from .query_expansion import aqe
from .rank import evaluate_rank
from .rerank import re_ranking
from .roc import evaluate_roc
from fastreid.utils import comm
from fastreid.utils.visualizer import Visualizer

logger = logging.getLogger(__name__)


class ReidEvaluator(DatasetEvaluator):
    def __init__(self, cfg, num_query, output_dir=None, thres=0.5):
        self.cfg = cfg
        self._num_query = num_query
        self._output_dir = output_dir

        self.features = []
        self.pids = []
        self.camids = []

        # For our metrics
        self._cpu_device = torch.device("cpu")
        self.thres = thres
        self.pred_logits = []
        self.gt_labels = []

    def reset(self):
        self.features = []
        self.pids = []
        self.camids = []

        # Our Metrics
        self.pred_logits = []
        self.gt_labels = []

    def process(self, inputs, outputs):
        self.pids.extend(inputs["targets"])
        self.camids.extend(inputs["camids"])
        self.features.append(outputs.cpu())

        # Our metrics

        #self.gt_labels.extend(inputs["targets"])
        #self.pred_logits.extend(outputs.cpu())

        self.gt_labels.extend(inputs["targets"].to(self._cpu_device))
        self.pred_logits.extend(outputs.to(self._cpu_device, torch.float32))

    @staticmethod
    def cal_dist(metric: str, query_feat: torch.tensor, gallery_feat: torch.tensor):
        assert metric in ["cosine", "euclidean"], "must choose from [cosine, euclidean], but got {}".format(metric)
        if metric == "cosine":
            dist = 1 - torch.mm(query_feat, gallery_feat.t())
        else:
            m, n = query_feat.size(0), gallery_feat.size(0)
            xx = torch.pow(query_feat, 2).sum(1, keepdim=True).expand(m, n)
            yy = torch.pow(gallery_feat, 2).sum(1, keepdim=True).expand(n, m).t()
            dist = xx + yy
            dist.addmm_(query_feat, gallery_feat.t(), beta=1, alpha=-2)
            dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist.cpu().numpy()

    # Our metric function
    @staticmethod
    def get_attr_metrics(gt_labels, pred_logits, thres):

        pred_labels = copy.deepcopy(pred_logits)
        pred_labels[pred_logits < thres] = 0
        pred_labels[pred_logits >= thres] = 1

        # Compute label-based metric
        overlaps = pred_labels * gt_labels
        correct_pos = overlaps.sum(axis=0)
        real_pos = gt_labels.sum(axis=0)
        inv_overlaps = (1 - pred_labels) * (1 - gt_labels)
        correct_neg = inv_overlaps.sum(axis=0)
        real_neg = (1 - gt_labels).sum(axis=0)

        # Compute instance-based accuracy
        pred_labels = pred_labels.astype(bool)
        gt_labels = gt_labels.astype(bool)
        intersect = (pred_labels & gt_labels).astype(float)
        union = (pred_labels | gt_labels).astype(float)

        ins_acc = (intersect.sum(axis=0) / union.sum(axis=0)).mean()
        ins_prec = (intersect.sum(axis=0) / pred_labels.astype(float).sum(axis=0)).mean()
        ins_rec = (intersect.sum(axis=0) / gt_labels.astype(float).sum(axis=0)).mean()
        ins_f1 = (2 * ins_prec * ins_rec) / (ins_prec + ins_rec)

        term1 = correct_pos / real_pos
        term2 = correct_neg / real_neg
        label_mA_verbose = (term1 + term2) * 0.5
        label_mA = label_mA_verbose.mean()

        results = OrderedDict()
        results["Accu"] = ins_acc
        results["Prec"] = ins_prec
        results["Recall"] = ins_rec
        results["F1"] = ins_f1
        results["mA"] = label_mA
        return results

    @staticmethod
    def get_attr_metrics_new(gt_labels, pred_logits, thres):

        eps = 1e-20

        pred_labels = copy.deepcopy(pred_logits)
        pred_labels[pred_logits < thres] = 0
        pred_labels[pred_logits >= thres] = 1



        # Compute label-based metric
        overlaps = pred_labels * gt_labels

        correct_pos = overlaps.sum(axis=0)
        real_pos = gt_labels.sum(axis=0)
        inv_overlaps = (1 - pred_labels) * (1 - gt_labels)
        correct_neg = inv_overlaps.sum(axis=0)
        real_neg = (1 - gt_labels).sum(axis=0)

        # Compute instance-based accuracy
        pred_labels = pred_labels.astype(bool)
        gt_labels = gt_labels.astype(bool)
        intersect = (pred_labels & gt_labels).astype(float)
        union = (pred_labels | gt_labels).astype(float)
        #ins_acc = (intersect.sum(axis=1) / (union.sum(axis=1) + eps)).mean()
        #ins_prec = (intersect.sum(axis=1) / (pred_labels.astype(float).sum(axis=1) + eps)).mean()
        #ins_rec = (intersect.sum(axis=1) / (gt_labels.astype(float).sum(axis=1) + eps)).mean()
        #ins_f1 = (2 * ins_prec * ins_rec) / (ins_prec + ins_rec + eps)


        term1 = correct_pos / (real_pos + eps)
        term2 = correct_neg / (real_neg + eps)
        label_mA_verbose = (term1 + term2) * 0.5
        label_mA = label_mA_verbose.mean()

        results = OrderedDict()
        #results["Accu"] = ins_acc * 100
        #results["Prec"] = ins_prec * 100
        #results["Recall"] = ins_rec * 100
        #results["F1"] = ins_f1 * 100
        results["mA"] = label_mA * 100
        results["metric"] = label_mA * 100
        return results

    def evaluate(self):
        if comm.get_world_size() > 1:
            comm.synchronize()
            features = comm.gather(self.features)
            features = sum(features, [])

            pids = comm.gather(self.pids)
            pids = sum(pids, [])

            camids = comm.gather(self.camids)
            camids = sum(camids, [])

            # Our metrics
            pred_logits = comm.gather(self.pred_logits)
            pred_logits = sum(pred_logits, [])

            gt_labels = comm.gather(self.gt_labels)
            gt_labels = sum(gt_labels, [])

            # fmt: off
            if not comm.is_main_process(): return {}
            # fmt: on
        else:
            features = self.features
            pids = self.pids
            camids = self.camids

            # Our code
            pred_logits = self.pred_logits
            gt_labels = self.gt_labels

        # Our code
        #pred_logits = torch.cat(pred_logits, dim=0)
        pred_logits = torch.stack(pred_logits, dim=0).numpy()
        gt_labels = torch.stack(gt_labels, dim=0).numpy()
        thres = self.thres
        pred_logits = pred_logits[..., 0]



        features = torch.cat(features, dim=0)
        # query feature, person ids and camera ids
        query_features = features[:self._num_query]
        query_pids = np.asarray(pids[:self._num_query])
        query_camids = np.asarray(camids[:self._num_query])



        # gallery features, person ids and camera ids
        gallery_features = features[self._num_query:]
        gallery_pids = np.asarray(pids[self._num_query:])
        gallery_camids = np.asarray(camids[self._num_query:])

        self._results = OrderedDict()

        if self.cfg.TEST.AQE.ENABLED:
            logger.info("Test with AQE setting")
            qe_time = self.cfg.TEST.AQE.QE_TIME
            qe_k = self.cfg.TEST.AQE.QE_K
            alpha = self.cfg.TEST.AQE.ALPHA
            query_features, gallery_features = aqe(query_features, gallery_features, qe_time, qe_k, alpha)

        if self.cfg.TEST.METRIC == "cosine":
            query_features = F.normalize(query_features, dim=1)
            gallery_features = F.normalize(gallery_features, dim=1)

        dist = self.cal_dist(self.cfg.TEST.METRIC, query_features, gallery_features)

        if self.cfg.TEST.RERANK.ENABLED:
            logger.info("Test with rerank setting")
            k1 = self.cfg.TEST.RERANK.K1
            k2 = self.cfg.TEST.RERANK.K2
            lambda_value = self.cfg.TEST.RERANK.LAMBDA
            q_q_dist = self.cal_dist(self.cfg.TEST.METRIC, query_features, query_features)
            g_g_dist = self.cal_dist(self.cfg.TEST.METRIC, gallery_features, gallery_features)
            re_dist = re_ranking(dist, q_q_dist, g_g_dist, k1, k2, lambda_value)
            query_features = query_features.numpy()
            gallery_features = gallery_features.numpy()
            cmc, all_AP, all_INP = evaluate_rank(re_dist, query_features, gallery_features,
                                                 query_pids, gallery_pids, query_camids,
                                                 gallery_camids, use_distmat=True)
        else:
            query_features = query_features.numpy()
            gallery_features = gallery_features.numpy()
            cmc, all_AP, all_INP = evaluate_rank(dist, query_features, gallery_features,
                                                 query_pids, gallery_pids, query_camids, gallery_camids,
                                                 use_distmat=False)
        mAP = np.mean(all_AP)
        mINP = np.mean(all_INP)
        for r in [1, 5, 10]:
            self._results['Rank-{}'.format(r)] = cmc[r - 1]
        self._results['mAP'] = mAP
        self._results['mINP'] = mINP

        if self.cfg.TEST.ROC_ENABLED:
            scores, labels = evaluate_roc(dist, query_features, gallery_features,
                                          query_pids, gallery_pids, query_camids, gallery_camids)
            fprs, tprs, thres = metrics.roc_curve(labels, scores)

            # Plotting the ROC
            fig = plt.figure()
            # Plot the ROC curve
            roc_auc = metrics.auc(fprs, tprs)
            plt.title('Receiver Operating Characteristic')
            plt.plot(fprs, tprs, 'b', label='AUC = %0.2f' % roc_auc)
            plt.legend(loc='lower right')
            plt.plot([0, 1], [0, 1], 'r--')
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            plt.show()
            filepath = os.path.join('plots', "roc_eval.jpg")
            plt.savefig(filepath)

            for fpr in [1e-4, 1e-3, 1e-2]:
                ind = np.argmin(np.abs(fprs - fpr))
                self._results["TPR@FPR={:.0e}".format(fpr)] = tprs[ind]
        return copy.deepcopy(self._results)

