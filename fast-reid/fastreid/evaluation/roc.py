# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""

import warnings

import faiss
import numpy as np

try:
    from .rank_cylib.roc_cy import evaluate_roc_cy

    IS_CYTHON_AVAI = True
except ImportError:
    IS_CYTHON_AVAI = False
    warnings.warn(
        'Cython roc evaluation (very fast so highly recommended) is '
        'unavailable, now use python evaluation.'
    )


def evaluate_roc_py(distmat, q_feats, g_feats, q_pids, g_pids, q_camids, g_camids):
    r"""
    Evaluation with ROC curve. The function compares query features and gallery features to calculate the cosine distance
    between the two. The key is that the gallery images that have the same pid and camid as the query image will be
    discarded.

    Args:
        distmat (np.ndarray): cosine distance matrix
        q_feats (np.ndarray): features of query images
        g_feats (np.ndarray): features of gallery images
        q_pids (np.ndarray): pids (person IDs) of query images
        g_pids (np.ndarray): pids (person IDs) of gallery images
        q_camids (np.ndarray): camids (camera IDs) of query images
        g_camids (np.ndarray): camids (camera IDs) of gallery images

    Returns:
        scores (np.ndarray): cosine distances between query and gallery features, where the distances between query images
        and the same camera view are discarded.
        labels (np.ndarray): 0 if the distance is between positive images (images with the same pid), and 1 if the
        distance is between negative images (images with different pids)
    """
    # Get the number of query images and number of gallery images
    num_q, num_g = distmat.shape
    # Get the dimension of the features
    dim = q_feats.shape[1]

    # Create a faiss index with cosine distance as the metric
    index = faiss.IndexFlatL2(dim)
    # Add the gallery features to the index
    index.add(g_feats)

    # Search for the nearest neighbors of each query feature
    _, indices = index.search(q_feats, k=num_g)
    # Create a matrix that indicates whether a query and gallery image pair has the same person ID
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    pos = []
    neg = []
    # Loop through all the query images
    for q_idx in range(num_q):
        # Get the person ID and camera ID of the current query image
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # Remove the gallery images that have the same person ID and camera ID as the query image
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)
        raw_cmc = matches[q_idx][keep]
        sort_idx = order[keep]

        # Get the cosine distance between the query and gallery images
        q_dist = distmat[q_idx]
        # Get the indices of the gallery images that have the same person ID as the query image
        ind_pos = np.where(raw_cmc == 1)[0]
        # Add the cosine distances between the query and positive gallery images to the `pos` list
        pos.extend(q_dist[sort_idx[ind_pos]])

        # Get the indices of the gallery images that have a different person ID from the query image
        ind_neg = np.where(raw_cmc == 0)[0]
        # Add the cosine distances between the query and negative gallery images to the `neg` list
        neg.extend(q_dist[sort_idx[ind_neg]])

    # Concatenate the positive and negative distances into a single list
    scores = np.hstack((pos, neg))

    # Create a label list indicating whether each distance is positive or negative
    labels = np.hstack((np.zeros(len(pos)), np.ones(len(neg))))
    return scores, labels


def evaluate_roc(
        distmat,
        q_feats,
        g_feats,
        q_pids,
        g_pids,
        q_camids,
        g_camids,
        use_cython=True
):
    """Evaluates CMC rank.
    Args:
        distmat (numpy.ndarray): distance matrix of shape (num_query, num_gallery).
        q_pids (numpy.ndarray): 1-D array containing person identities
            of each query instance.
        g_pids (numpy.ndarray): 1-D array containing person identities
            of each gallery instance.
        q_camids (numpy.ndarray): 1-D array containing camera views under
            which each query instance is captured.
        g_camids (numpy.ndarray): 1-D array containing camera views under
            which each gallery instance is captured.
        use_cython (bool, optional): use cython code for evaluation. Default is True.
            This is highly recommended as the cython code can speed up the cmc computation
            by more than 10x. This requires Cython to be installed.
    """
    if use_cython and IS_CYTHON_AVAI:
        return evaluate_roc_cy(distmat, q_feats, g_feats, q_pids, g_pids, q_camids, g_camids)
    else:
        return evaluate_roc_py(distmat, q_feats, g_feats, q_pids, g_pids, q_camids, g_camids)
