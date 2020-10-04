from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray
import numpy as np
from keras.utils import Sequence

from keras import backend as K

import warnings

import cv2
import numpy

from keras import backend as K

warnings.filterwarnings("ignore", category=DeprecationWarning)

from keras.applications.vgg16 import preprocess_input

def warn(*args, **kwargs):
    pass


def preProcess(dataLocation, imageSize, grayScale):


    data = cv2.imread(dataLocation)
    data = numpy.array(cv2.resize(data, imageSize))

    if grayScale:
       data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
       data = numpy.expand_dims(data, axis=2)

    # else:
    #     data = numpy.swapaxes(data, 1, 2)
    #     data = numpy.swapaxes(data, 0, 1)

    data = data.astype('float16')

    data /= 255

    data = numpy.array(data)

    return data

class ArousalValenceGenerator(Sequence):

    def __init__(self, image_filenames, labels, batch_size, imageSize, grayScale=False):
        self.image_filenames, self.labels = np.array(image_filenames), np.array(labels)
        self.batch_size = batch_size
        self.imageSize = imageSize
        self.grayScale = grayScale

    def __len__(self):

        #print "Len:", np.ceil(len(self.image_filenames) / float(self.batch_size))
        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))

    def __getitem__(self, idx):

        # print "List index:", idx
        # print "Total:", len(self.image_filenames)
        # print "load from:", idx * self.batch_size, " : ", (idx + 1) * self.batch_size
        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        # batch = np.array([
        #     preProcess(file_name, self.imageSize, self.grayScale)
        #     for file_name in batch_x]), [np.array(batch_y[:, 0]), np.array(batch_y[:, 1])]

        batch = np.array([
            preProcess(file_name, self.imageSize, self.grayScale)
            for file_name in batch_x]), np.array(batch_y[:, 0])

        #
        # batch = np.array([
        #     preprocess_input(cv2.imread(file_name))
        #     for file_name in batch_x]), [np.array(batch_y[:, 0]), np.array(batch_y[:, 1])]

        #
        # batch = np.array([
        #     preprocess_input(cv2.imread(file_name))
        #     for file_name in batch_x]), [np.array(batch_y[:, 0])]

        return batch