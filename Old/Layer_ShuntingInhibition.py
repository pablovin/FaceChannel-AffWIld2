#More info: Hing, F., Tivive, C., & Bouzerdoum, A. (2006, August). A shunting inhibitory convolutional neural network for gender classification. In Pattern Recognition, 2006. ICPR 2006. 18th International Conference on (Vol. 4, pp. 421-424). IEEE.

from keras import backend as K
from keras.engine.topology import Layer
import numpy

from keras import backend as K

import tensorflow as tf

class ShuntingInhibition(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(ShuntingInhibition, self).__init__(**kwargs)

    def build(self, input_shape):

        # if K.image_dim_ordering() == "th":
        initialDecay = numpy.full((1,1,64), 0.5, dtype="float32")
        # else:
        #     initialDecay = numpy.full((1, 1, input_shape[0][3]), 0.5)

        self._inhibitionDecay = tf.Variable(initial_value=initialDecay, trainable=False)
        # K.variable(initialDecay)
        # self.trainable_weights = tf.Variable(initial_value=initialDecay, trainable=True)
        # [self._inhibitionDecay]

        super(ShuntingInhibition, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        v_c, v_c_inhibit = x

        result = (v_c / (self._inhibitionDecay + v_c_inhibit))
        return result

    def compute_output_shape(self, input_shape):
        return input_shape[0]