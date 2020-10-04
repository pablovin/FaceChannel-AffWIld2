import cv2
import numpy
from tensorflow.python.keras.utils.data_utils import Sequence
import tensorflow as tf

from imgaug import augmenters as iaa

GENERATORTYPE = {'EXP_FaceChannel': 'EXP_FaceChannel',
                 'EXP_ResNet50v2': 'EXP_ResNet50v2'
                 }

"""Factory"""

def getGenerator(generatorType, samples, labels, batchSize, inputShape, dataAugmentation=False, sequence=False):

    if generatorType == GENERATORTYPE["EXP_FaceChannel"]:
        if dataAugmentation:

            dataGenerator = EXP_GeneratorFaceChannel(samples, labels, batchSize, inputShape, augmentation=seq, sequence=sequence)

            return dataGenerator

        else:
            return EXP_GeneratorFaceChannel(samples, labels, batchSize, inputShape, sequence=sequence)
    elif generatorType == GENERATORTYPE["EXP_ResNet50v2"]:
        if dataAugmentation:

            dataGenerator = EXP_GeneratorResNet50V2(samples, labels, batchSize, inputShape, augmentation=seq, sequence=sequence)

            return dataGenerator

        else:
            return EXP_GeneratorResNet50V2(samples, labels, batchSize, inputShape, sequence=sequence)


"""Utils"""
def preProcess(dataLocation, grayScale, imageSize):

    data = cv2.imread(dataLocation)
    # print ("location:" + str(dataLocation))
    data = numpy.array(cv2.resize(data, imageSize))

    if grayScale:
       data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
       data = numpy.expand_dims(data, axis=2)

    data = numpy.array(data, dtype='float16')
    # data = (data / 255.0) * 2 - 1
    data = (data / 255.0)

    return data


def preProcessSequences(dataLocations, grayScale, imageSize):

    images = []
    for dataLocation in dataLocations:
        # print ("DataLocation:" + str(dataLocation))
        data = cv2.imread(dataLocation)
        # data = numpy.array(cv2.resize(data, imageSize))
        # input ("here")
        # if grayScale:
        #     data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
        #     data = numpy.expand_dims(data, axis=2)

        data = numpy.array(data, dtype='float16') / 255.0

        images.append(data)

    images = numpy.array(images)

    # print ("SequenceSize:" + str(images.shape))
    # input("Here")
    return images



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
class EXP_GeneratorFaceChannel(Sequence):

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
                self.preprocess(file_name, self.grayScale, self.imageSize)
                for file_name in batch_x]), batch_y

        else:
            # print ("Augmenting!!!")
            batch = numpy.array([
                self.augmentation.augment_image(self.preprocess(file_name, self.grayScale, self.imageSize))
                for file_name in batch_x]), batch_y

        # print ("Batch shape:" + str(batch[0].shape))
        # input("here")
        return batch

