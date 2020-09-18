from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Input
from keras.layers import Dense, Dropout, Flatten, Lambda
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import LeakyReLU
from keras.optimizers import Adam, Adamax, SGD
from keras.callbacks import ModelCheckpoint,  EarlyStopping , ReduceLROnPlateau
from keras import regularizers

from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization

from keras.models import load_model

from keras.models import Model

from keras.callbacks import  Callback

import numpy
import copy


from Layer_ShuntingInhibition import ShuntingInhibition

from GeneratorAffWild2 import ArousalValenceGenerator

from metrics import fbeta_score, recall, precision, rmse, ccc

import keras


class CustomModelCheckpoint(Callback):

    previousAcc = 0

    def __init__(self, model, path):

        # This is the argument that will be modify by fit_generator
        # self.model = model
        self.path = path

        # We set the model (non multi gpu) under an other name
        self.model_for_saving = model

    def on_epoch_end(self, epoch, logs=None):


        loss = logs['arousal_output_ccc']
        if loss > self.previousAcc:
            print("-------------------------------------------------------\n")
            print("-- IMPROVED --\n")
            print("Arousal CCC "+str(self.previousAcc)+ " --> "+str(loss)+"\n")
            print("Saving model to : {}".format(self.path.format(epoch=epoch, val_loss=loss))+"\n")
            print("-------------------------------------------------------\n")
            self.previousAcc = loss

            # Here we save the original one

            #self.model_for_saving.save_weights(self.path.format(epoch=epoch, val_loss=loss), overwrite=True)
            self.model.save(self.path.format(epoch=epoch, val_loss=loss)+".hdf5")
        else:
            print ("-------------------------------------------------------\n")
            print("-- not IMPROVED --\n")
            print("Arousal CCC " + str(self.previousAcc) + " --> " + str(loss) + "\n")
            print ("-------------------------------------------------------\n")

def loadModel(directory):
   from tensorflow import keras

   model = keras.models.load_model(directory,
                             custom_objects={'fbeta_score': fbeta_score, 'rmse': rmse,
                                             'recall': recall, 'precision': precision,
                                             'ccc': ccc})


   model.summary()

   return model

def buildModel(inputShape, numberOfOutputs):

    # keras.backend.set_image_data_format("channels_first")

    nch = 256

    # inputShape = numpy.array((1, 64, 64)).astype(numpy.int32)
    inputLayer = Input(shape=inputShape, name="Vision_Network_Input")

    # Conv1 and 2
    conv1 = Conv2D(int(nch / 4), (3, 3), padding="same", kernel_initializer="glorot_uniform", activation="relu",
                   name="Vision_conv1")(inputLayer)

    conv2 = Conv2D(int(nch / 4), (3, 3), padding="same", kernel_initializer="glorot_uniform", activation="relu",
                   name="Vision_conv2")(conv1)

    mp1 = MaxPooling2D(pool_size=(2, 2))(conv2)
    drop1 = Dropout(0.25)(mp1)

    # Conv 3 and 4
    conv3 = Conv2D(int(nch / 2), (3, 3), padding="same", kernel_initializer="glorot_uniform", activation="relu",
                   name="Vision_conv3")(drop1)

    conv4 = Conv2D(int(nch / 2), (3, 3), padding="same", kernel_initializer="glorot_uniform", activation="relu",
                   name="Vision_conv4")(conv3)

    mp2 = MaxPooling2D(pool_size=(2, 2))(conv4)
    drop2 = Dropout(0.25)(mp2)

    # Conv 5 and 6 and 7
    conv5 = Conv2D(int(nch / 2), (3, 3), padding="same", kernel_initializer="glorot_uniform", activation="relu",
                   name="Vision_conv5")(drop2)

    conv6 = Conv2D(int(nch / 2), (3, 3), padding="same", kernel_initializer="glorot_uniform", activation="relu",
                   name="Vision_conv6")(conv5)

    conv7 = Conv2D(int(nch / 2), (3, 3), padding="same", kernel_initializer="glorot_uniform", activation="relu",
                   name="Vision_conv7")(conv6)

    mp3 = MaxPooling2D(pool_size=(2, 2))(conv7)
    drop3 = Dropout(0.25)(mp3)
    # Conv 8 and 9 and 10
    conv8 = Conv2D(nch, (3, 3), padding="same", kernel_initializer="glorot_uniform", activation="relu",
                   name="Vision_conv8")(drop3)

    conv9 = Conv2D(nch, (3, 3), padding="same", kernel_initializer="glorot_uniform", activation="relu",
                   name="conv9")(conv8)

    conv10 = Conv2D(nch, (3, 3), padding="same", kernel_initializer="glorot_uniform", activation="relu",
                    name="conv10")(conv9)

    # conv10_inhibition = Conv2D(nch, (3, 3), padding="same", kernel_initializer="glorot_uniform", activation="relu",
    #                 name="conv10_Inhibition")(conv9)
    #
    # inhibition = ShuntingInhibition(name="inhibitoryLayer")([conv10, conv10_inhibition])

    mp4 = MaxPooling2D(pool_size=(2, 2))(conv10)
    drop4 = Dropout(0.25)(mp4)

    flatten = Flatten()(drop4)

    dense = Dense(200, activation="relu", name="denseLayer")(flatten)
    drop5 = Dropout(0.5)(dense)

    arousal_output = Dense(units=1, activation='linear', name='arousal_output')(drop5)
    valence_output = Dense(units=1, activation='linear', name='valence_output')(drop5)

    model = Model(inputs=inputLayer, outputs=[arousal_output, valence_output])
    return model

def evaluate(model, validationSamples, imgSize):

    batchSize = 32

    optimizer = SGD(learning_rate=0.01, momentum=0.0, nesterov=False)

    model.compile(loss={'arousal_output':'mean_squared_error', 'valence_output':'mean_squared_error'},
                  optimizer=optimizer,
                  metrics=[ccc])


    validationGenerator = ArousalValenceGenerator(validationSamples[0], validationSamples[1], batchSize, imgSize,
                                                  grayScale=False)

    scores = model.evaluate_generator(validationGenerator, len(validationSamples))  # 1514 testing images
    print("Scores = ", scores)


def train (model, trainSamples, testSamples, validationSamples, imgSize, experimentFolder):


    # for layer in model.layers:
    #     layer.trainable = False
    #
    # model.get_layer(name="denseLayer").trainable = True
    # model.get_layer(name="arousal_output").trainable = True
    # model.get_layer(name="valence_output").trainable = True
    model.summary()

    batchSize = 64
    epoches = 40

    optimizer = SGD(learning_rate=0.01, momentum=0.0, nesterov=False)

    trainGenerator = ArousalValenceGenerator(trainSamples[0], trainSamples[1], batchSize, imgSize, grayScale=False)

    validationGenerator = ArousalValenceGenerator(validationSamples[0],validationSamples[1],  batchSize, imgSize, grayScale=False)

    # testGenerator = CategoricalGenerator(testSamples[0], testSamples[1], batchSize, imgSize, grayScale=True)

    model.compile(loss={'arousal_output': 'mean_squared_error', 'valence_output': 'mean_squared_error'},
                  optimizer=optimizer,
                  metrics=[ccc])

    filepath = experimentFolder + "/weights.best.hdf5"
    checkpointsString = experimentFolder + '/weights.{epoch:02d}-{val_loss:.2f}.hdf5'

    checkPoint = CustomModelCheckpoint(model, checkpointsString)

    # checkPoint = ModelCheckpoint(filepath, monitor='val_categorical_accuracy',
    #                             verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    #

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001, verbose=1)

    # print "Steps per epoch:", len(dataPointsTrain.dataX) // self.batchSize

    history_callback = model.fit_generator(generator=trainGenerator,
                                                steps_per_epoch=len(trainSamples[0]) // batchSize,
                                                epochs=epoches,
                                                verbose=1,
                                                validation_data=validationGenerator,
                                                validation_steps=len(validationSamples[0]) // batchSize,
                                                use_multiprocessing=True,
                                                workers=30,
                                                max_queue_size=3591,
                                                callbacks=[checkPoint, reduce_lr],

                                                )