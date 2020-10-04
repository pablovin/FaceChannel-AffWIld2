"""
This file describes the data generators used to train and evaluate the CPC model.
"""

import numpy as np
from skimage.util import view_as_windows
import cv2
import skimage.color

GENERATORTYPE = {'CPCGenerator_AV': 'CPCGenerator_AV'
                 }

"""
Factory
"""

def getGenerator(generatorType, samples, labels, batchSize, n_negatives,
                 label_dim_mul=None, neg_same_ratio=0.75):

    augment_image_fn = augment_images_mnist
    augment_crop_fn = augment_crops_mnist

    if generatorType == GENERATORTYPE['CPCGenerator_AV']:
        return NCEGenerator(samples, labels, batchSize, n_negatives, augment_image_fn, augment_crop_fn,
                 label_dim_mul, neg_same_ratio)

"""
Utils
"""
def preProcess(dataLocation):

    data = cv2.imread(dataLocation)

    data = np.array(cv2.resize(data, (64,64)))

    data = np.array(data, dtype='float16')

    data = (data / 255.0) * 2 - 1

    return data


def augment_crops_mnist(crops):
    """ Performs cropping and HSV/HUE color augmentation in image crops. """

    def change_hsv(patch, h, s, v):

        # Convert
        patch_hsv = skimage.color.rgb2hsv(rgb=patch)

        # H Channel
        patch_hsv[:, :, 0] += h % 1.0
        patch_hsv[:, :, 0] %= 1.0

        # S Channel
        patch_hsv[:, :, 1] = np.clip(patch_hsv[:, :, 1] + s, 0, 1)

        # V Channel
        patch_hsv[:, :, 2] = np.clip(patch_hsv[:, :, 2] + v, 0, 1)

        # Convert back
        patch = skimage.color.hsv2rgb(hsv=patch_hsv)
        return patch

    # Pad
    crops_shape = crops.shape
    n = crops_shape[-3]
    p = crops_shape[2] // 8  # paper is //16
    crops = crops.reshape((-1,) + crops_shape[-3:])
    crops_pad = np.pad(crops, ((0, 0), (p, p), (p, p), (0, 0)), 'reflect')

    # Crop
    for i in range(crops_pad.shape[0]):
        crop = np.copy(crops_pad[i, ...])
        x = np.random.randint(0, p * 2 + 1)
        y = np.random.randint(0, p * 2 + 1)
        crop = crop[x:x + n, y:y + n, :]

        # Change color hue
        h = np.random.uniform(-1, 1)
        s = np.random.uniform(-0.5, 0.5)
        v = np.random.uniform(-0.5, 0.5)
        crop = change_hsv(crop * 0.5 + 0.5, h, s, v) * 2 - 1

        crops[i, ...] = crop

    # Reshape
    crops = crops.reshape(crops_shape)

    return crops


def augment_images_mnist(images):
    """ Performs simple image augmentation (cropping). """

    # Pad
    x_shape = images.shape
    n = x_shape[1]
    p = n // 6
    images_pad = np.pad(images, ((0, 0), (p, p), (p, p), (0, 0)), 'reflect')

    # Crop
    for i in range(images.shape[0]):
        x_i = np.random.randint(0, p * 2 + 1)
        y_i = np.random.randint(0, p * 2 + 1)
        im = images_pad[i, x_i:x_i + n, y_i:y_i + n, :]
        images[i, :, :, :] = im

    return images


"""
Generators
"""
class NCEGenerator(object):

    def __init__(self, fileNames, labels, batch_size, n_negatives, augment_image_fn, augment_crop_fn,
                 label_dim_mul=None, neg_same_ratio=0.75):

        """
        Noise-contrastive-estimation sample generator. It performs several functions:
        - Augments the images.
        - Extracts overlapping crops from each image.
        - Augments the crops.
        - Extracts negative crops.
        - Provide labels.

        :param x_path: path to numpy file containing images in uint8 format and dimensions [samples, x, y, ch].
        :param y_path: path to numpy file containing labels in uint8 format and dimensions [samples, ].
        :param batch_size: number of samples per mini-batch.
        :param n_classes: total number of classes (required to produce one-hot encoding).
        :param n_negatives: number of negative crops to extract. If zero, no negative crops are extracted
        and the iterator returns tuples of crops and image labels (useful for classification).
        :param augment_image_fn: augmentation function applied to batch of images in uint8 format and
        dimensions [n_samples, x, y, ch].
        :param augment_crop_fn: augmentation function applied to crops of image in float [-1, +1] format and
        dimensions [n_crops, n_crops, x, y, ch].
        :param label_dim_mul: how many times to repeat the labels (useful for multiple prediction directions).
        :param neg_same_ratio: percentage of negative crops that are sourced from the same image (vs other
        images in the batch).
        """

        # Set params
        # self.x_path = fileNames
        # self.y_path = labels
        self.batch_size = batch_size
        self.augment_crop_fn = augment_crop_fn
        self.n_negatives = n_negatives
        self.label_dim_mul = label_dim_mul
        self.neg_same_ratio = neg_same_ratio

        # Image generator
        self.image_generator = ImageGenerator(
            fileNames=fileNames,
            labels=labels,
            batch_size=batch_size,
            augment_fn=augment_image_fn
        )

        self.n_batches = len(self.image_generator)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self.n_batches

    def get_image_gen(self):
        return self.image_generator

    def next(self):
        """
        If n_negatives is a positive number:
            Returns ([crops, neg_crops], y) with dimensions:
                crops: [batch, crop_row, crop_col, x, y, ch]
                neg_crops: [batch, crop_row, crop_col, n_neg, x, y, ch]
                y: [batch, label_dim_mul, n_neg]
        Else:
            Returns (crops, labels) with dimensions:
                crops: [batch, crop_row, crop_col, x, y, ch]
                labels: [batch, n_classes]
        """

        np.random.seed()  # crucial for multiprocessing (otherwise, all processes will be initialized with same
                          # seed, thus same sequence of patches)

        # Get data
        x, labels = next(self.image_generator)

        # Extract crops from images: [batch, x, y, ch] to [batch, crop_row, crop_col, x, y, ch]
        crops = []
        for image in x:
            crops.append(image_to_crops(image, patch_size=16, stride=8, augment_fn=self.augment_crop_fn))
        crops = np.stack(crops, axis=0)

        # Collect negative samples (crops)
        if self.n_negatives > 0:

            # Some negative crops are taken from the same image, and some from other images in the batch
            n_same = int(self.n_negatives * self.neg_same_ratio)
            n_all = self.n_negatives - n_same
            neg_crops_all = get_negative_patches_all(crops, n_neg=n_all)
            neg_crops_same = get_negative_patches_same(crops, n_neg=n_same)

            # Concatenate crops from same and other images
            neg_crops = np.concatenate([neg_crops_all, neg_crops_same], axis=3)

            # Labels (1 for the first element everywhere)
            y = np.eye(self.n_negatives + 1)[np.zeros(crops.shape[0], dtype='uint8')].astype('uint8')

            # Repeat labels in case of multiple prediction directions
            y = np.stack([y] * self.label_dim_mul, axis=1)
        else:
            neg_crops = []
            y = []

        if self.n_negatives > 0:
            return ([crops, neg_crops], y)
        else:
            return (crops, labels)


class ImageGenerator(object):

    def __init__(self, fileNames, labels, batch_size, augment_fn=None):
        """
        Iterator that yields batches of images and labels from disk sampled randomly.

        :param x_path: path to numpy file containing images in uint8 format and dimensions [samples, x, y, ch].
        :param y_path: path to numpy file containing labels in uint8 format and dimensions [samples, ].
        :param batch_size: number of samples per mini-batch.
        :param n_classes: total number of classes (required to produce one-hot encoding).
        :param augment_fn: augmentation function applied to batch of images in uint8 format.
        """

        # Params
        self.x_file_names = fileNames
        self.y_labels = labels
        self.batch_size = batch_size
        self.augment_fn = augment_fn

        # # Load data
        # self.y = np.load(y_path)
        # self.x = None

        self.n_samples = len(self.y_labels)
        self.n_batches = self.n_samples // batch_size

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self.n_batches

    def next(self):
        """
        Images in [-1, +1] format and dimensions [samples, x, y, ch], labels in one-hot code and
        dimensions [samples, classes].

        :return: tuple (images, labels).
        """

        # # Load data
        # if self.x is None:
        #     self.x = np.load(self.x_path)

        # Sample
        idx = np.random.choice(len(self.x_file_names), self.batch_size, replace=False)
        x = self.x_file_names[idx, ...]
        y = self.y_labels[idx, ...]

        #Load and pre-process
        x =  np.array([
                preProcess(file_name)
                for file_name in x])

        # Augment
        if self.augment_fn is not None:

            x = self.augment_fn(x)


        # Format
        # x = (x / 255.0) * 2 - 1
        # y = np.eye(self.n_classes)[y]

        return x, y


def image_to_crops(image, patch_size, stride, augment_fn=None):
    """
    Returns a grid of patches from the given image, taking into account patch size and stride.

    :param image: input image with dimensions (x, y, ch).
    :param patch_size: size of crop/patch.
    :param stride: size of stride in pixels. For example, 50% overlapping means stride=patch_size//2.
    :param augment_fn: augmentation function applied to crops of image in float [-1, +1] format and
        dimensions [n_crops, n_crops, x, y, ch].
    :return: crops with dimensions [n_crops, n_crops, x, y, ch]
    """

    # Get patches
    crops = view_as_windows(image, (patch_size, patch_size, 3), (stride, stride, 1)).squeeze()

    # Augment
    if augment_fn is not None:
        crops = augment_fn(crops)

    return crops


def get_negative_patches_all(crops, n_neg):
    """
    Sample random negative patches (crops) from any image.

    :param crops: input crops with dim [batch, crop_row, crop_col, x, y, ch].
    :param n_neg: number of negative crops to extract.
    :return: output crops with dim [batch, crop_row, crop_col, n_neg, x, y, ch].
    """

    # Prepare variables
    n_batch, n_rows, n_cols, n_height, n_width, n_ch = crops.shape
    crops = np.copy(crops).reshape((n_batch * n_rows * n_cols, n_height, n_width, n_ch))
    neg_crops = np.zeros((n_batch * n_rows * n_cols, n_height, n_width, n_ch, n_neg))

    # For each image in the batch
    for i in range(crops.shape[0]):

        # Sample from anywhere except the correct crop
        p = (np.arange(crops.shape[0]) != i) / np.sum(np.arange(crops.shape[0]) != i)
        idx = np.random.choice(crops.shape[0], n_neg, replace=False, p=p)
        negs = crops[idx, ...].transpose((1, 2, 3, 0))
        neg_crops[i, ...] = negs

    neg_crops = neg_crops.reshape((n_batch, n_rows, n_cols, n_height, n_width, n_ch, n_neg))

    return neg_crops.transpose((0, 1, 2, 6, 3, 4, 5))


def get_negative_patches_same(crops, n_neg):
    """
    Sample random negative patches from the same image. These negative patches are more difficult to distinguish
    from the correct one since they are sourced from the same image (look similar).

    :param crops: input crops with dim [batch, crop_row, crop_col, x, y, ch].
    :param n_neg: number of negative crops to extract.
    :return: output crops with dim [batch, crop_row, crop_col, n_neg, x, y, ch].
    """

    # Prepare variables
    n_batch, n_rows, n_cols, n_height, n_width, n_ch = crops.shape
    n_patches = n_rows * n_cols
    crops = np.copy(crops).reshape((n_batch * n_rows * n_cols, n_height, n_width, n_ch))
    neg_crops = np.zeros((n_batch * n_rows * n_cols, n_height, n_width, n_ch, n_neg))

    # For each image in the batch
    for i in range(crops.shape[0]):

        # Sample from any location in the current image except the correct patch
        p = np.zeros(crops.shape[0])
        p[(i // n_patches) * n_patches: (i // n_patches) * n_patches + n_patches] = 1
        p[i] = 0
        m = np.sum(p)
        p = p / np.sum(p)
        idx = np.random.choice(crops.shape[0], n_neg, replace=True if n_neg > m else False, p=p)
        negs = crops[idx, ...].transpose((1, 2, 3, 0))
        neg_crops[i, ...] = negs

    neg_crops = neg_crops.reshape((n_batch, n_rows, n_cols, n_height, n_width, n_ch, n_neg))

    return neg_crops.transpose((0, 1, 2, 6, 3, 4, 5))

