from __future__ import division, print_function, absolute_import
import math
import random
from collections import deque
import torch
from PIL import Image
from torchvision.transforms import (
    Resize, Compose, ToTensor, Normalize, ColorJitter, RandomHorizontalFlip,
    RandomResizedCrop, RandomApply, RandomGrayscale, GaussianBlur
)


class Random2DTranslation(object):
    """Randomly translates the input image with a probability.

    Specifically, given a predefined shape (height, width), the input is first
    resized with a factor of 1.125, leading to (height*1.125, width*1.125), then
    a random crop is performed. Such operation is done with a probability.

    Args:
        height (int): target image height.
        width (int): target image width.
        p (float, optional): probability that this operation takes place.
            Default is 0.5.
        interpolation (int, optional): desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, height, width, p=0.5, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.p = p
        self.interpolation = interpolation

    def __call__(self, img):
        if random.uniform(0, 1) > self.p:
            return img.resize((self.width, self.height), self.interpolation)

        new_width, new_height = int(round(self.width * 1.125)
                                    ), int(round(self.height * 1.125))
        resized_img = img.resize((new_width, new_height), self.interpolation)
        x_maxrange = new_width - self.width
        y_maxrange = new_height - self.height
        x1 = int(round(random.uniform(0, x_maxrange)))
        y1 = int(round(random.uniform(0, y_maxrange)))
        croped_img = resized_img.crop(
            (x1, y1, x1 + self.width, y1 + self.height)
        )
        return croped_img


class RandomErasing(object):
    """Randomly erases an image patch.

    Origin: `<https://github.com/zhunzhong07/Random-Erasing>`_

    Reference:
        Zhong et al. Random Erasing Data Augmentation.

    Args:
        probability (float, optional): probability that this operation takes place.
            Default is 0.5.
        sl (float, optional): min erasing area.
        sh (float, optional): max erasing area.
        r1 (float, optional): min aspect ratio.
        mean (list, optional): erasing value.
    """

    def __init__(
        self,
        probability=0.5,
        sl=0.02,
        sh=0.4,
        r1=0.3,
        mean=[0.4914, 0.4822, 0.4465]
    ):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img


class ColorAugmentation(object):
    """Randomly alters the intensities of RGB channels.

    Reference:
        Krizhevsky et al. ImageNet Classification with Deep ConvolutionalNeural
        Networks. NIPS 2012.

    Args:
        p (float, optional): probability that this operation takes place.
            Default is 0.5.
    """

    def __init__(self, p=0.5):
        self.p = p
        self.eig_vec = torch.Tensor(
            [
                [0.4009, 0.7192, -0.5675],
                [-0.8140, -0.0045, -0.5808],
                [0.4203, -0.6948, -0.5836],
            ]
        )
        self.eig_val = torch.Tensor([[0.2175, 0.0188, 0.0045]])

    def _check_input(self, tensor):
        assert tensor.dim() == 3 and tensor.size(0) == 3

    def __call__(self, tensor):
        if random.uniform(0, 1) > self.p:
            return tensor
        alpha = torch.normal(mean=torch.zeros_like(self.eig_val)) * 0.1
        quatity = torch.mm(self.eig_val * alpha, self.eig_vec)
        tensor = tensor + quatity.view(3, 1, 1)
        return tensor


class RandomPatch(object):
    """Random patch data augmentation.

    There is a patch pool that stores randomly extracted pathces from person images.
    
    For each input image, RandomPatch
        1) extracts a random patch and stores the patch in the patch pool;
        2) randomly selects a patch from the patch pool and pastes it on the
           input (at random position) to simulate occlusion.

    Reference:
        - Zhou et al. Omni-Scale Feature Learning for Person Re-Identification. ICCV, 2019.
        - Zhou et al. Learning Generalisable Omni-Scale Representations
          for Person Re-Identification. TPAMI, 2021.
    """

    def __init__(
        self,
        prob_happen=0.5,
        pool_capacity=50000,
        min_sample_size=100,
        patch_min_area=0.01,
        patch_max_area=0.5,
        patch_min_ratio=0.1,
        prob_rotate=0.5,
        prob_flip_leftright=0.5,
    ):
        self.prob_happen = prob_happen

        self.patch_min_area = patch_min_area
        self.patch_max_area = patch_max_area
        self.patch_min_ratio = patch_min_ratio

        self.prob_rotate = prob_rotate
        self.prob_flip_leftright = prob_flip_leftright

        self.patchpool = deque(maxlen=pool_capacity)
        self.min_sample_size = min_sample_size

    def generate_wh(self, W, H):
        area = W * H
        for attempt in range(100):
            target_area = random.uniform(
                self.patch_min_area, self.patch_max_area
            ) * area
            aspect_ratio = random.uniform(
                self.patch_min_ratio, 1. / self.patch_min_ratio
            )
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < W and h < H:
                return w, h
        return None, None

    def transform_patch(self, patch):
        if random.uniform(0, 1) > self.prob_flip_leftright:
            patch = patch.transpose(Image.FLIP_LEFT_RIGHT)
        if random.uniform(0, 1) > self.prob_rotate:
            patch = patch.rotate(random.randint(-10, 10))
        return patch

    def __call__(self, img):
        W, H = img.size # original image size

        # collect new patch
        w, h = self.generate_wh(W, H)
        if w is not None and h is not None:
            x1 = random.randint(0, W - w)
            y1 = random.randint(0, H - h)
            new_patch = img.crop((x1, y1, x1 + w, y1 + h))
            self.patchpool.append(new_patch)

        if len(self.patchpool) < self.min_sample_size:
            return img

        if random.uniform(0, 1) > self.prob_happen:
            return img

        # paste a randomly selected patch on a random position
        patch = random.sample(self.patchpool, 1)[0]
        patchW, patchH = patch.size
        x1 = random.randint(0, W - patchW)
        y1 = random.randint(0, H - patchH)
        patch = self.transform_patch(patch)
        img.paste(patch, (x1, y1))

        return img

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]

    def __repr__(self):
        format_string  = self.__class__.__name__ + 'with base transform: \n'
        format_string += str(self.base_transform)
        return format_string

def build_transforms(
    height,
    width,
    transforms='random_flip',
    norm_mean=[0.485, 0.456, 0.406],
    norm_std=[0.229, 0.224, 0.225],
    **kwargs
):
    """Builds train and test transform functions.

    Args:
        height (int): target image height.
        width (int): target image width.
        transforms (str or list of str, optional): transformations applied to model training.
            Default is 'random_flip'.
        norm_mean (list or None, optional): normalization mean values. Default is ImageNet means.
        norm_std (list or None, optional): normalization standard deviation values. Default is
            ImageNet standard deviation values.
    """
    if transforms is None:
        transforms = []

    if isinstance(transforms, str):
        transforms = [transforms]

    if not isinstance(transforms, list):
        raise ValueError(
            'transforms must be a list of strings, but found to be {}'.format(
                type(transforms)
            )
        )

    if len(transforms) > 0:
        transforms = [t.lower() for t in transforms]

    if norm_mean is None or norm_std is None:
        norm_mean = [0.485, 0.456, 0.406] # imagenet mean
        norm_std = [0.229, 0.224, 0.225] # imagenet std
    normalize = Normalize(mean=norm_mean, std=norm_std)

    print('Building train transforms ...')
    transform_tr = []

    print('+ resize to {}x{}'.format(height, width))
    transform_tr += [Resize((height, width))]

    if 'random_flip' in transforms:
        print('+ random flip')
        transform_tr += [RandomHorizontalFlip()]

    if 'random_crop' in transforms:
        print(
            '+ random crop (enlarge to {}x{} and '
            'crop {}x{})'.format(
                int(round(height * 1.125)), int(round(width * 1.125)), height,
                width
            )
        )
        transform_tr += [Random2DTranslation(height, width)]

    if 'random_patch' in transforms:
        print('+ random patch')
        transform_tr += [RandomPatch()]

    if 'color_jitter' in transforms:
        print('+ color jitter')
        transform_tr += [
            ColorJitter(brightness=0.2, contrast=0.15, saturation=0, hue=0)
        ]

    print('+ to torch tensor of range [0, 1]')
    transform_tr += [ToTensor()]

    print('+ normalization (mean={}, std={})'.format(norm_mean, norm_std))
    transform_tr += [normalize]

    if 'random_erase' in transforms:
        print('+ random erase')
        transform_tr += [RandomErasing(mean=norm_mean)]

    transform_tr = Compose(transform_tr)

    print('Building test transforms ...')
    print('+ resize to {}x{}'.format(height, width))
    print('+ to torch tensor of range [0, 1]')
    print('+ normalization (mean={}, std={})'.format(norm_mean, norm_std))

    transform_te = Compose([
        Resize((height, width)),
        ToTensor(),
        normalize,
    ])

    return transform_tr, transform_te

def get_costum_transformer(height=256, width=128, two_crop=True):
    # Normalize the mean and standard deviation values
    mean = [0.3525, 0.3106, 0.3140]
    std = [0.2660, 0.2522, 0.2505]
    normalizer = Normalize(mean=mean, std=std)

    # Initialize augmentation list with RandomResizedCrop
    augmentation = [RandomResizedCrop((height, width), scale=(0.2, 1.))]

    # Create a dictionary of augmentation arguments
    augmentation_dictionary = {'moco':{'cj': 1.0, 'cj-s': 0.4, 'gs': 0.2,}}

    # Get the augmentation arguments for the momentum contrast model
    augmentation_arguments = augmentation_dictionary['moco']
    color_jitter_p = augmentation_arguments.get('cj', 0.)
    color_jitter_s = augmentation_arguments.get('cj-s', 0.4)

    # Add ColorJitter to the augmentation list if color_jitter_p is between 0 and 1
    if 0. < color_jitter_p < 1.:
        augmentation.append(RandomApply([ColorJitter(color_jitter_s, color_jitter_s, color_jitter_s, 0.1)],  color_jitter_p))
    # Add ColorJitter to the augmentation list if color_jitter_p is equal to 1
    elif color_jitter_p == 1:
        augmentation.append(ColorJitter(color_jitter_s, color_jitter_s, color_jitter_s, color_jitter_s))

    # Add RandomGrayscale to the augmentation list if gray_scale_p is between 0 and 1
    gray_scale_p = augmentation_arguments.get('gs', 0.)
    if 0. < gray_scale_p <= 1.:
        augmentation.append(RandomGrayscale(p=gray_scale_p))

    # Add RandomApply of GaussianBlur to the augmentation list if gaussian_blur_p is between 0 and 1
    gaussian_blur_p = augmentation_arguments.get('gb', 0.)
    if 0. < gaussian_blur_p < 1.:
        augmentation.append(RandomApply([GaussianBlur([.1,.2])], p=gaussian_blur_p))

    # Add RandomHorizontalFlip and ToTensor to the augmentation list
    augmentation.append(RandomHorizontalFlip())
    augmentation.append(ToTensor())
    augmentation.append(normalizer)

    # Get the probability of erasing
    random_erase_p = augmentation_arguments.get('re', 0.)
    # Get the maximum erasing area
    random_erase_s = augmentation_arguments.get('re-s', 0.4)

    # Add RandomErasing to the augmentation list if random_erase_p is between 0 and 1
    if 0. < random_erase_p < 1.:
        augmentation.append((RandomErasing(probability=random_erase_p, sh=random_erase_s)))

    # Create a transformer by composing all the augmentations in the list
    transformer = Compose(augmentation)

    # If two_crop is True, wrap the transformer with TwoCropsTransform
    if two_crop:
        transformer = TwoCropsTransform(transformer)
    return transformer

def get_validation_transformer(height=256, width=128):
    # Define the mean and standard deviation values for normalization
    mean = [0.3525, 0.3106, 0.3140]
    std = [0.2660, 0.2522, 0.2505]

    # Create a normalizer object using the mean and standard deviation values
    normalizer = Normalize(mean=mean, std=std)

    # Create a transformer using the Resize, ToTensor, and normalizer operations
    transformer = Compose([Resize((height, width), interpolation=3), ToTensor(), normalizer])

    return transformer