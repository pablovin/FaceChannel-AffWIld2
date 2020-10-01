import cv2
import numpy
from tensorflow.python.keras.utils.data_utils import Sequence
import tensorflow as tf

from imgaug import augmenters as iaa

GENERATORTYPE = {'Arousal_FaceChannel': 'Arousal_FaceChannel',
                 'Arousal_ResNet50v2': 'Arousal_ResNet50v2'
                 }

"""Factory"""

def getGenerator(generatorType, samples, labels, batchSize, inputShape, dataAugmentation=False, sequence=False):

    if generatorType == GENERATORTYPE["Arousal_FaceChannel"]:
        if dataAugmentation:

            dataGenerator = ArousalGeneratorFaceChannel(samples, labels, batchSize, inputShape, augmentation=seq, sequence=sequence)

            return dataGenerator

        else:
            return ArousalGeneratorFaceChannel(samples, labels, batchSize, inputShape, sequence=sequence)
    elif generatorType == GENERATORTYPE["Arousal_ResNet50v2"]:
        if dataAugmentation:

            dataGenerator = ArousalGeneratorResNet50V2(samples, labels, batchSize, inputShape, augmentation=seq, sequence=sequence)

            return dataGenerator

        else:
            return ArousalGeneratorResNet50V2(samples, labels, batchSize, inputShape, sequence=sequence)


"""Utils"""
def preProcess(dataLocation, grayScale):

    data = cv2.imread(dataLocation)

    if grayScale:
       data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
       data = numpy.expand_dims(data, axis=2)

    data = numpy.array(data, dtype='float16') / 255.0

    return data


def preProcessSequences(dataLocations, grayScale):

    images = []
    for dataLocation in dataLocations:
        # print ("DataLocation:" + str(dataLocation))
        data = cv2.imread(dataLocation)
        # input ("here")
        if grayScale:
            data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
            data = numpy.expand_dims(data, axis=2)

        data = numpy.array(data, dtype='float16') / 255.0

        images.append(data)

    images = numpy.array(images)

    # print ("Shape:" + str(images.shape))
    # input("here")
    return images


# st = lambda aug: iaa.Sometimes(0.5, aug)
#
# seq = iaa.Sequential([
#     iaa.Fliplr(0.5), # horizontally flip
#     # st(iaa.Add((-10, 10), per_channel=0.5)),
#     # st(iaa.Multiply((0.5, 1.5), per_channel=0.5)),
#     # st(iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5)),
#     st(iaa.Affine(
#         scale={"x": (0.9, 1.10), "y": (0.9, 1.10)},
#         # scale images to 80-120% of their size, individually per axis
#         translate_px={"x": (-5, 5), "y": (-5, 5)},  # translate by -16 to +16 pixels (per axis)
#         rotate=(-10, 10),  # rotate by -45 to +45 degrees
#         shear=(-3, 3),  # shear by -16 to +16 degrees
#         order=3,  # use any of scikit-image's interpolation methods
#         cval=(0.0, 1.0),  # if mode is constant, use a cval between 0 and 1.0
#         mode="constant"
#     )),
# ], random_order=True)


seq = iaa.Sequential([
    iaa.Fliplr(0.5),  # horizontally flip
    # sometimes(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05), per_channel=0.5)),
    iaa.OneOf([
        iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
        iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
        iaa.GaussianBlur(sigma=(0.0, 1.0)),
        iaa.Affine(rotate=(-10, 10), translate_percent={"x": (-0.25, 0.25)}, mode='symmetric', cval=(0)),

    ]),
], random_order=True)


"""Generators"""

class ArousalGeneratorResNet50V2(Sequence):

    def __init__(self, image_filenames, labels, batch_size, imageSize, augmentation=None, sequence=False):

        self.image_filenames, self.labels = image_filenames, labels
        self.batch_size = batch_size
        self.imageSize = (imageSize[0], imageSize[1])

        if imageSize[2]== 1:
            self.grayScale = True
        else:
            self.grayScale = False

        self.augmentation = augmentation


        if sequence:
            self.preprocess = preProcessSequences
        else:
            self.preprocess = preProcess


    def __len__(self):
        return int(numpy.ceil(len(self.image_filenames) / float(self.batch_size)))

    def __getitem__(self, idx):

        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        if self.augmentation == None:
            batch = numpy.array([
                preProcess(file_name, self.grayScale)
                for file_name in batch_x]), batch_y

        else:
            batch = numpy.array([
                self.augmentation.augment_image(tf.keras.applications.resnet.preprocess_input(cv2.imread(file_name)))
                for file_name in batch_x]), batch_y

        return batch

class ArousalGeneratorFaceChannel(Sequence):

    def __init__(self, image_filenames, labels, batch_size, imageSize, augmentation=None, sequence=False):

        self.image_filenames, self.labels = image_filenames, labels
        self.batch_size = batch_size
        self.imageSize = (imageSize[0], imageSize[1])

        if imageSize[2]== 1:
            self.grayScale = True
        else:
            self.grayScale = False

        self.augmentation = augmentation

        if sequence:
            self.preprocess = preProcessSequences
        else:
            self.preprocess = preProcess


        # print ("Sequence:" + str(sequence))
        # input("here")

    def __len__(self):
        return int(numpy.ceil(len(self.image_filenames) / float(self.batch_size)))

    def __getitem__(self, idx):

        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        if self.augmentation == None:
            batch = numpy.array([
                self.preprocess(file_name, self.grayScale)
                for file_name in batch_x]), batch_y

        else:
            # print ("Augmenting!!!")
            batch = numpy.array([
                self.augmentation.augment_image(self.preprocess(file_name, self.grayScale))
                for file_name in batch_x]), batch_y

        return batch

