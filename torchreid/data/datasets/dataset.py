from __future__ import division, print_function, absolute_import
import copy
import pickle

import lmdb
import numpy as np
import os.path as osp
import tarfile
import zipfile
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset as dtst

from torchreid.utils import read_image, download_url, mkdir_if_missing


class Dataset(object):
    """An abstract class representing a Dataset.

    This is the base class for ``ImageDataset`` and ``VideoDataset``.

    Args:
        train (list): contains tuples of (img_path(s), pid, camid).
        query (list): contains tuples of (img_path(s), pid, camid).
        gallery (list): contains tuples of (img_path(s), pid, camid).
        transform: transform function.
        k_tfm (int): number of times to apply augmentation to an image
            independently. If k_tfm > 1, the transform function will be
            applied k_tfm times to an image. This variable will only be
            useful for training and is currently valid for image datasets only.
        mode (str): 'train', 'query' or 'gallery'.
        combineall (bool): combines train, query and gallery in a
            dataset for training.
        verbose (bool): show information.
    """

    # junk_pids contains useless person IDs, e.g. background,
    # false detections, distractors. These IDs will be ignored
    # when combining all images in a dataset for training, i.e.
    # combineall=True
    _junk_pids = []

    # Some datasets are only used for training, like CUHK-SYSU
    # In this case, "combineall=True" is not used for them
    _train_only = False

    def __init__(
        self,
        train,
        query,
        gallery,
        transform=None,
        k_tfm=1,
        mode='train',
        combineall=False,
        verbose=True,
        **kwargs
    ):
        # extend 3-tuple (img_path(s), pid, camid) to
        # 4-tuple (img_path(s), pid, camid, dsetid) by
        # adding a dataset indicator "dsetid"
        if len(train[0]) == 3:
            train = [(*items, 0) for items in train]
        if len(query[0]) == 3:
            query = [(*items, 0) for items in query]
        if len(gallery[0]) == 3:
            gallery = [(*items, 0) for items in gallery]

        self.train = train
        self.query = query
        self.gallery = gallery
        self.transform = transform
        self.k_tfm = k_tfm
        self.mode = mode
        self.combineall = combineall
        self.verbose = verbose

        self.num_train_pids = self.get_num_pids(self.train)
        self.num_train_cams = self.get_num_cams(self.train)
        self.num_datasets = self.get_num_datasets(self.train)

        if self.combineall:
            self.combine_all()

        if self.mode == 'train':
            self.data = self.train
        elif self.mode == 'query':
            self.data = self.query
        elif self.mode == 'gallery':
            self.data = self.gallery
        else:
            raise ValueError(
                'Invalid mode. Got {}, but expected to be '
                'one of [train | query | gallery]'.format(self.mode)
            )

        if self.verbose:
            self.show_summary()

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def __add__(self, other):
        """Adds two datasets together (only the train set)."""
        train = copy.deepcopy(self.train)

        for img_path, pid, camid, dsetid in other.train:
            pid += self.num_train_pids
            camid += self.num_train_cams
            dsetid += self.num_datasets
            train.append((img_path, pid, camid, dsetid))

        ###################################
        # Note that
        # 1. set verbose=False to avoid unnecessary print
        # 2. set combineall=False because combineall would have been applied
        #    if it was True for a specific dataset; setting it to True will
        #    create new IDs that should have already been included
        ###################################
        if isinstance(train[0][0], str):
            return ImageDataset(
                train,
                self.query,
                self.gallery,
                transform=self.transform,
                mode=self.mode,
                combineall=False,
                verbose=False
            )
        else:
            return VideoDataset(
                train,
                self.query,
                self.gallery,
                transform=self.transform,
                mode=self.mode,
                combineall=False,
                verbose=False,
                seq_len=self.seq_len,
                sample_method=self.sample_method
            )

    def __radd__(self, other):
        """Supports sum([dataset1, dataset2, dataset3])."""
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def get_num_pids(self, data):
        """Returns the number of training person identities.

        Each tuple in data contains (img_path(s), pid, camid, dsetid).
        """
        pids = set()
        for items in data:
            pid = items[1]
            pids.add(pid)
        return len(pids)

    def get_num_cams(self, data):
        """Returns the number of training cameras.

        Each tuple in data contains (img_path(s), pid, camid, dsetid).
        """
        cams = set()
        for items in data:
            camid = items[2]
            cams.add(camid)
        return len(cams)

    def get_num_datasets(self, data):
        """Returns the number of datasets included.

        Each tuple in data contains (img_path(s), pid, camid, dsetid).
        """
        dsets = set()
        for items in data:
            dsetid = items[3]
            dsets.add(dsetid)
        return len(dsets)

    def show_summary(self):
        """Shows dataset statistics."""
        pass

    def combine_all(self):
        """Combines train, query and gallery in a dataset for training."""
        if self._train_only:
            return

        combined = copy.deepcopy(self.train)

        # relabel pids in gallery (query shares the same scope)
        g_pids = set()
        for items in self.gallery:
            pid = items[1]
            if pid in self._junk_pids:
                continue
            g_pids.add(pid)
        pid2label = {pid: i for i, pid in enumerate(g_pids)}

        def _combine_data(data):
            for img_path, pid, camid, dsetid in data:
                if pid in self._junk_pids:
                    continue
                pid = pid2label[pid] + self.num_train_pids
                combined.append((img_path, pid, camid, dsetid))

        _combine_data(self.query)
        _combine_data(self.gallery)

        self.train = combined
        self.num_train_pids = self.get_num_pids(self.train)

    def download_dataset(self, dataset_dir, dataset_url):
        """Downloads and extracts dataset.

        Args:
            dataset_dir (str): dataset directory.
            dataset_url (str): url to download dataset.
        """
        if osp.exists(dataset_dir):
            return

        if dataset_url is None:
            raise RuntimeError(
                '{} dataset needs to be manually '
                'prepared, please follow the '
                'document to prepare this dataset'.format(
                    self.__class__.__name__
                )
            )

        print('Creating directory "{}"'.format(dataset_dir))
        mkdir_if_missing(dataset_dir)
        fpath = osp.join(dataset_dir, osp.basename(dataset_url))

        print(
            'Downloading {} dataset to "{}"'.format(
                self.__class__.__name__, dataset_dir
            )
        )
        download_url(dataset_url, fpath)

        print('Extracting "{}"'.format(fpath))
        try:
            tar = tarfile.open(fpath)
            tar.extractall(path=dataset_dir)
            tar.close()
        except:
            zip_ref = zipfile.ZipFile(fpath, 'r')
            zip_ref.extractall(dataset_dir)
            zip_ref.close()

        print('{} dataset is ready'.format(self.__class__.__name__))

    def check_before_run(self, required_files):
        """Checks if required files exist before going deeper.

        Args:
            required_files (str or list): string file name(s).
        """
        if isinstance(required_files, str):
            required_files = [required_files]

        for fpath in required_files:
            if not osp.exists(fpath):
                raise RuntimeError('"{}" is not found'.format(fpath))

    def __repr__(self):
        num_train_pids = self.get_num_pids(self.train)
        num_train_cams = self.get_num_cams(self.train)

        num_query_pids = self.get_num_pids(self.query)
        num_query_cams = self.get_num_cams(self.query)

        num_gallery_pids = self.get_num_pids(self.gallery)
        num_gallery_cams = self.get_num_cams(self.gallery)

        msg = '  ----------------------------------------\n' \
              '  subset   | # ids | # items | # cameras\n' \
              '  ----------------------------------------\n' \
              '  train    | {:5d} | {:7d} | {:9d}\n' \
              '  query    | {:5d} | {:7d} | {:9d}\n' \
              '  gallery  | {:5d} | {:7d} | {:9d}\n' \
              '  ----------------------------------------\n' \
              '  items: images/tracklets for image/video dataset\n'.format(
                  num_train_pids, len(self.train), num_train_cams,
                  num_query_pids, len(self.query), num_query_cams,
                  num_gallery_pids, len(self.gallery), num_gallery_cams
              )

        return msg

    def _transform_image(self, tfm, k_tfm, img0):
        """Transforms a raw image (img0) k_tfm times with
        the transform function tfm.
        """
        img_list = []

        for k in range(k_tfm):
            img_list.append(tfm(img0))

        img = img_list
        if len(img) == 1:
            img = img[0]

        return img


class ImageDataset(Dataset):
    """A base class representing ImageDataset.

    All other image datasets should subclass it.

    ``__getitem__`` returns an image given index.
    It will return ``img``, ``pid``, ``camid`` and ``img_path``
    where ``img`` has shape (channel, height, width). As a result,
    data in each batch has shape (batch_size, channel, height, width).
    """

    def __init__(self, train, query, gallery, **kwargs):
        super(ImageDataset, self).__init__(train, query, gallery, **kwargs)

    def __getitem__(self, index):
        img_path, pid, camid, dsetid = self.data[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self._transform_image(self.transform, self.k_tfm, img)
        item = {
            'img': img,
            'pid': pid,
            'camid': camid,
            'impath': img_path,
            'dsetid': dsetid
        }
        return item

    def show_summary(self):
        num_train_pids = self.get_num_pids(self.train)
        num_train_cams = self.get_num_cams(self.train)

        num_query_pids = self.get_num_pids(self.query)
        num_query_cams = self.get_num_cams(self.query)

        num_gallery_pids = self.get_num_pids(self.gallery)
        num_gallery_cams = self.get_num_cams(self.gallery)

        print('=> Loaded {}'.format(self.__class__.__name__))
        print('  ----------------------------------------')
        print('  subset   | # ids | # images | # cameras')
        print('  ----------------------------------------')
        print(
            '  train    | {:5d} | {:8d} | {:9d}'.format(
                num_train_pids, len(self.train), num_train_cams
            )
        )
        print(
            '  query    | {:5d} | {:8d} | {:9d}'.format(
                num_query_pids, len(self.query), num_query_cams
            )
        )
        print(
            '  gallery  | {:5d} | {:8d} | {:9d}'.format(
                num_gallery_pids, len(self.gallery), num_gallery_cams
            )
        )
        print('  ----------------------------------------')


class VideoDataset(Dataset):
    """A base class representing VideoDataset.

    All other video datasets should subclass it.

    ``__getitem__`` returns an image given index.
    It will return ``imgs``, ``pid`` and ``camid``
    where ``imgs`` has shape (seq_len, channel, height, width). As a result,
    data in each batch has shape (batch_size, seq_len, channel, height, width).
    """

    def __init__(
        self,
        train,
        query,
        gallery,
        seq_len=15,
        sample_method='evenly',
        **kwargs
    ):
        super(VideoDataset, self).__init__(train, query, gallery, **kwargs)
        self.seq_len = seq_len
        self.sample_method = sample_method

        if self.transform is None:
            raise RuntimeError('transform must not be None')

    def __getitem__(self, index):
        img_paths, pid, camid, dsetid = self.data[index]
        num_imgs = len(img_paths)

        if self.sample_method == 'random':
            # Randomly samples seq_len images from a tracklet of length num_imgs,
            # if num_imgs is smaller than seq_len, then replicates images
            indices = np.arange(num_imgs)
            replace = False if num_imgs >= self.seq_len else True
            indices = np.random.choice(
                indices, size=self.seq_len, replace=replace
            )
            # sort indices to keep temporal order (comment it to be order-agnostic)
            indices = np.sort(indices)

        elif self.sample_method == 'evenly':
            # Evenly samples seq_len images from a tracklet
            if num_imgs >= self.seq_len:
                num_imgs -= num_imgs % self.seq_len
                indices = np.arange(0, num_imgs, num_imgs / self.seq_len)
            else:
                # if num_imgs is smaller than seq_len, simply replicate the last image
                # until the seq_len requirement is satisfied
                indices = np.arange(0, num_imgs)
                num_pads = self.seq_len - num_imgs
                indices = np.concatenate(
                    [
                        indices,
                        np.ones(num_pads).astype(np.int32) * (num_imgs-1)
                    ]
                )
            assert len(indices) == self.seq_len

        elif self.sample_method == 'all':
            # Samples all images in a tracklet. batch_size must be set to 1
            indices = np.arange(num_imgs)

        else:
            raise ValueError(
                'Unknown sample method: {}'.format(self.sample_method)
            )

        imgs = []
        for index in indices:
            img_path = img_paths[int(index)]
            img = read_image(img_path)
            if self.transform is not None:
                img = self.transform(img)
            img = img.unsqueeze(0) # img must be torch.Tensor
            imgs.append(img)
        imgs = torch.cat(imgs, dim=0)

        item = {'img': imgs, 'pid': pid, 'camid': camid, 'dsetid': dsetid}

        return item

    def show_summary(self):
        num_train_pids = self.get_num_pids(self.train)
        num_train_cams = self.get_num_cams(self.train)

        num_query_pids = self.get_num_pids(self.query)
        num_query_cams = self.get_num_cams(self.query)

        num_gallery_pids = self.get_num_pids(self.gallery)
        num_gallery_cams = self.get_num_cams(self.gallery)

        print('=> Loaded {}'.format(self.__class__.__name__))
        print('  -------------------------------------------')
        print('  subset   | # ids | # tracklets | # cameras')
        print('  -------------------------------------------')
        print(
            '  train    | {:5d} | {:11d} | {:9d}'.format(
                num_train_pids, len(self.train), num_train_cams
            )
        )
        print(
            '  query    | {:5d} | {:11d} | {:9d}'.format(
                num_query_pids, len(self.query), num_query_cams
            )
        )
        print(
            '  gallery  | {:5d} | {:11d} | {:9d}'.format(
                num_gallery_pids, len(self.gallery), num_gallery_cams
            )
        )
        print('  -------------------------------------------')

class CostumDataset(dtst):
    def __init__(self, data_dir, key_path, transform=None):
        super(CostumDataset, self).__init__()
        self.data_dir = data_dir
        self.key_path = key_path
        self.transform = transform

        if not osp.exists(self.data_dir):
            raise IOError('dataset directiory: {} is not exists'.format(
                self.data_dir))
        self.load_dataset_infos()
        self.env = None

    def load_dataset_infos(self):
        if not osp.exists(self.key_path):
            raise IOError('key file: {} not exists'.format(
                self.key_path))
        with open(self.key_path, 'rb') as f:
            data = pickle.load(f)
        self.keys = data
        if 'pids' in data:
            self.labels = np.array(data['pids'], np.int)
        elif 'vids' in data:
            self.labels = np.array(data['vids'], np.int)
        else:
            self.labels = np.zeros(len(self.keys), np.int)
        self.num_cls = len(set(self.labels))

    def __len__(self):
        return len(self.keys)

    def _init_lmdb(self):
        self.env = lmdb.open(self.data_dir, readonly=True, lock=False,
                             readahead=False, meminit=False)

    def __getitem__(self, index):
        if self.env is None:
            self._init_lmdb()

        key = self.keys[index]
        label = self.labels[index]

        with self.env.begin(write=False) as txn:
            buf = txn.get(key.encode('ascii'))
        img_flat = np.frombuffer(buf, dtype=np.uint8)
        im = cv2.imdecode(img_flat, cv2.IMREAD_COLOR)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            im = Image.fromarray(im)
            im1 =self.transform(im)
            if self.strong_transform is not None:
                im2 = self.strong_transform(im)
            else:
                im / 255
        else:
            im1 = im / 255
            if self.strong_transform(im):
                im2 = self.strong_transform(im)
            else:
                im2 = im / 255
        return im1, im2, label

class StandardDataset(dtst):

    def __init__(self, root='data', dataset_name='market', mode='train', transform=None):
        self.root = root
        self.dataset_name = dataset_name
        self.mode = mode
        self.transform = transform
        self.dataset_directory = osp.join(self.root, self.dataset_name)
        self.train_directory = osp.join(self.dataset_directory, 'bounding_box_train')
        self.query_directory = osp.join(self.dataset_directory, 'query')
        self.gallery_directory = osp.join(self.dataset_directory, 'bounding_box_test')
        if self.mode == 'train':
            self.items, self.num_pids, self_cams = self.preprocess(self.train_directory)
        else:
            self.query, self.num_query_pids, self.num_query_cams = self.preprocess(self.query_directory, False, is_query=True)
            self.gallery, self.num_gallery_pids, self.num_gallery_cams = self.preprocess(self.gallery_directory, False, is_query=False)
            self.items = self.query + self.gallery
            self.num_pids = max(self.num_gallery_pids, self.num_query_pids)
            self.num_cams = max(self.num_query_cams, self.num_gallery_cams)

    def preprocess(self, path, is_query=None):
        all_pids, all_cids = {}, {}
        results, file_paths = [], []
        for