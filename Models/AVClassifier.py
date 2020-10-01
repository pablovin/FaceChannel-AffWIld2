from tensorflow import keras
from Metrics.metrics import ccc
from keras.models import Sequential, Input
from keras.layers import Dense, Dropout, Flatten, Lambda, TimeDistributed, LSTM
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.optimizers import Adam, Adamax, SGD
from keras.layers.core import Activation
from keras.models import Model

import tensorflow as tf

from Models.Layer_ShuntingInhibition import ShuntingInhibition

import os
from datetime import datetime

MODELTYPE = {'Arousal_Frame_FaceChannel': 'Arousal_Frame_FaceChannel',
             'Arousal_Frame_ResNet50v2': 'Arousal_Frame_ResNet50v2',

             'Arousal_Frame_FaceChannel_Optmizer': 'Arousal_Frame_FaceChannel_Optmizer',

              'Arousal_Sequence_FaceChannel': 'Arousal_Sequence_FaceChannel',
             }


"""
Factory to get the right model
"""

def getModel(inputShape, modelType, params=[]):

    if modelType == MODELTYPE["Arousal_Frame_FaceChannel"]:
        return arousalFrameFaceChannelModel(inputShape)
    elif modelType == MODELTYPE["Arousal_Frame_ResNet50v2"]:
        return arousalFrameResNet50(inputShape)
    elif modelType == MODELTYPE["Arousal_Sequence_FaceChannel"]:
        return arousalSequenceFaceChannelModel(inputShape)

    elif modelType == MODELTYPE["Arousal_Frame_FaceChannel_Optmizer"]:
        return arousalFrameFaceChannelModel_Optmizer(inputShape, params)



"""
Utils
"""

def evaluate(model, validationGenerator, batchSize):

    scores = model.evaluate(validationGenerator, batch_size=batchSize)
    print("Scores = ", scores)
    return scores

def loadModel(directory):

   model = keras.models.load_model(directory,
                             custom_objects={'ccc': ccc})

   # for layer in model.layers:
   #     layer.trainable = False
   #
   # model.get_layer(name="denseLayer").trainable = True
   # model.get_layer(name="arousal_output").trainable = True


   print ("----------------")
   print ("Loaded:" + str(directory))
   model.summary()
   print ("----------------")

   return model

def createFolders(folder):

    if not os.path.exists(folder):
        os.mkdir(folder)

def train(model, experimentFolder, trainGenerator, validationGenerator, batchSize = 64, epoches=10, params=[], verbose=1):

    if len(params) > 0:
        # denseLayer, initialLR, decay, momentum, nesterov, batchSize = 500, 0.015476808651646383, True, 0.7408493385691893, True, 1024
        denseLayer, initialLR, decay, momentum, nesterov, batchSize, smallNetwork, inhibition = params

    print("----------------")
    print("Training this model:")
    model.summary()
    print("----------------")

    """Create folders for training"""

    modelFolder = experimentFolder+"/Model"
    historyFolder = experimentFolder+"/History"
    tensorBoard = experimentFolder+"/TensorBoard"

    createFolders(experimentFolder)
    createFolders(modelFolder)
    createFolders(historyFolder)
    createFolders(tensorBoard)

    """Callbacks """
    callbacks = [
        # keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=1/5, patience=2, min_lr=1e-5, verbose=1),
        keras.callbacks.CSVLogger(filename=historyFolder + '/history.csv', separator=',', append=True),
        keras.callbacks.ModelCheckpoint(filepath=modelFolder, monitor='val_loss',
                                        save_best_only=True, mode='min'),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, mode='min'),
        keras.callbacks.TensorBoard(log_dir=tensorBoard, histogram_freq=0, write_graph=True, write_images=False)
    ]


    model.fit(x=trainGenerator, batch_size=batchSize,
              epochs= epoches, verbose = verbose, shuffle=True,
              validation_data= validationGenerator, max_queue_size=1024,
              callbacks=callbacks
              )

    if len(os.listdir(modelFolder)) > 0:
        model = loadModel(modelFolder)

    return model

"""
Models
"""


def arousalSequenceFaceChannelModel(inputShape):
    # keras.backend.set_image_data_format("channels_first")

    nch = 64

    # inputShape = numpy.array((1, 64, 64)).astype(numpy.int32)
    # inputLayer = Input(shape=inputShape, name="Vision_Network_Input")

    model = Sequential()

    # Conv1 and 2
    model.add(
        TimeDistributed(
            Conv2D(int(nch / 4), (3, 3), padding="same", kernel_initializer="glorot_uniform", activation="relu",
                   ), input_shape=(inputShape), name="Vision_conv1"
        )
    )

    model.add(
        TimeDistributed(
            Conv2D(int(nch / 4), (3, 3), padding="same", kernel_initializer="glorot_uniform", activation="relu",
                   ), name="Vision_conv2"
        )
    )

    model.add(
        TimeDistributed(
            MaxPooling2D(pool_size=(2, 2)), name="pool1"
        )
    )

    model.add(
        TimeDistributed(Dropout(0.5), name="drop1"
                        )
    )

    # Conv 3 and 4
    model.add(
        TimeDistributed(
            Conv2D(int(nch / 2), (3, 3), padding="same", kernel_initializer="glorot_uniform", activation="relu",
                   ), name="Vision_conv3"
        )
    )

    model.add(
        TimeDistributed(
            Conv2D(int(nch / 2), (3, 3), padding="same", kernel_initializer="glorot_uniform", activation="relu",
                   ), name="Vision_conv4"
        )
    )

    model.add(
        TimeDistributed(
            MaxPooling2D(pool_size=(2, 2)), name="pool2"
        )
    )

    model.add(
        TimeDistributed(Dropout(0.5), name="drop2"
                        )
    )


    # Flatten

    model.add(
        TimeDistributed(Flatten(), name="Flatten"
                        )
    )

    # RNN

    model.add(
        LSTM(10, activation='relu', return_sequences=False, name="Rnn_1")

    )

    model.add(
        Dense(10, activation='relu', name="dense_1")

    )
    model.add(
        Dropout(0.5, name="drop5")

    )

    model.build()

    arousal_output = Dense(units=1, activation='tanh', name='arousal_output')(model.output)

    model = Model(inputs=model.input, outputs=arousal_output)

    optimizer = Adam()

    model.compile(loss={'arousal_output': 'mean_squared_error'},
                  optimizer=optimizer,
                  metrics=[ccc])

    return model


def arousalFrameResNet50(inputShape):


    # new_input = Input(shape=inputShape)

    resNet = tf.keras.applications.ResNet50V2(include_top=False, pooling="max")

    dense = Dense(100, activation="relu", name="denseLayer")(resNet.output)
    drop5 = Dropout(0.5)(dense)

    arousal_output = Dense(units=1, activation='tanh', name='arousal_output')(drop5)

    model = Model(inputs=resNet.input, outputs=arousal_output)

    for layer in model.layers:
        layer.trainable = False

    model.get_layer(name="denseLayer").trainable = True
    model.get_layer(name="arousal_output").trainable = True

    optimizer = Adam()
    # optimizer = SGD(lr=0.1, momentum=0.9, decay=0.1 / 10)

    model.compile(loss={'arousal_output': 'mean_squared_error'},
                  optimizer=optimizer,
                  metrics=[ccc])




    return model

def arousalFrameFaceChannelModel(inputShape):

    """Model """
    nch = 64

    # inputShape = numpy.array((1, 64, 64)).astype(numpy.int32)
    inputLayer = Input(shape=inputShape, name="Vision_Network_Input")

    # Conv1 and 2
    conv1 = Conv2D(int(nch / 4), (3, 3), padding="same", kernel_initializer="glorot_uniform", activation=None,
                   name="Vision_conv1")(inputLayer)

    # conv1 = BatchNormalization()(conv1)

    conv1 = Activation("relu")(conv1)

    conv2 = Conv2D(int(nch / 4), (3, 3), padding="same", kernel_initializer="glorot_uniform", activation=None,
                   name="Vision_conv2")(conv1)
    # conv2 = BatchNormalization()(conv2)

    conv2 = Activation("relu")(conv2)

    mp1 = MaxPooling2D(pool_size=(2, 2))(conv2)
    drop1 = Dropout(0.5)(mp1)

    # Conv 3 and 4
    conv3 = Conv2D(int(nch / 2), (3, 3), padding="same", kernel_initializer="glorot_uniform", activation=None,
                   name="Vision_conv3")(drop1)
    # conv3 = BatchNormalization()(conv3)

    conv3 = Activation("relu")(conv3)

    conv4 = Conv2D(int(nch / 2), (3, 3), padding="same", kernel_initializer="glorot_uniform", activation=None,
                   name="Vision_conv4")(conv3)
    # conv4 = BatchNormalization()(conv4)

    conv4 = Activation("relu")(conv4)

    mp2 = MaxPooling2D(pool_size=(2, 2))(conv4)
    drop2 = Dropout(0.5)(mp2)


    #
    flatten = Flatten()(drop2)

    dense = Dense(500, activation="relu", name="denseLayer")(flatten)
    drop5 = Dropout(0.5)(dense)

    arousal_output = Dense(units=1, activation='tanh', name='arousal_output')(drop5)

    model = Model(inputs=inputLayer, outputs=arousal_output)


    """Training Parameters"""

    optimizer = SGD(learning_rate= 0.015, decay=0.1 / 10, momentum=0.74, nesterov=False)
    # optimizer = SGD(lr=0.1, momentum=0.9, decay=0.1 / 10)

    model.compile(loss={'arousal_output': 'mean_squared_error'},
                  optimizer=optimizer,
                  metrics=[ccc])


    return model



def arousalFrameFaceChannelModel_Optmizer(inputShape, params):

    # denseLayer, initialLR, decay, momentum, nesterov, batchSize = 500, 0.015476808651646383, True, 0.7408493385691893, True, 1024
    denseLayer, initialLR, decay, momentum, nesterov, batchSize, smallNetwork, inhibition = params

    """Model """
    nch = 64

    # inputShape = numpy.array((1, 64, 64)).astype(numpy.int32)
    inputLayer = Input(shape=inputShape, name="Vision_Network_Input")

    # Conv1 and 2
    conv1 = Conv2D(int(nch / 4), (3, 3), padding="same", kernel_initializer="glorot_uniform", activation=None,
                   name="Vision_conv1")(inputLayer)

    # conv1 = BatchNormalization()(conv1)

    conv1 = Activation("relu")(conv1)

    conv2 = Conv2D(int(nch / 4), (3, 3), padding="same", kernel_initializer="glorot_uniform", activation=None,
                   name="Vision_conv2")(conv1)
    # conv2 = BatchNormalization()(conv2)

    conv2 = Activation("relu")(conv2)

    mp1 = MaxPooling2D(pool_size=(2, 2))(conv2)
    drop1 = Dropout(0.5)(mp1)

    # Conv 3 and 4
    conv3 = Conv2D(int(nch / 2), (3, 3), padding="same", kernel_initializer="glorot_uniform", activation=None,
                   name="Vision_conv3")(drop1)
    # conv3 = BatchNormalization()(conv3)

    conv3 = Activation("relu")(conv3)

    conv4 = Conv2D(int(nch / 2), (3, 3), padding="same", kernel_initializer="glorot_uniform", activation=None,
                   name="Vision_conv4")(conv3)
    # conv4 = BatchNormalization()(conv4)

    conv4 = Activation("relu")(conv4)

    mp2 = MaxPooling2D(pool_size=(2, 2))(conv4)
    drop2 = Dropout(0.5)(mp2)


    # Conv 5 and 6 and 7
    conv5 = Conv2D(int(nch / 2), (3, 3), padding="same", kernel_initializer="glorot_uniform", activation=None,
                   name="Vision_conv5")(drop2)
    # conv5 = BatchNormalization()(conv5)
    conv5 = Activation("relu")(conv5)

    conv6 = Conv2D(int(nch / 2), (3, 3), padding="same", kernel_initializer="glorot_uniform", activation=None,
                   name="Vision_conv6")(conv5)
    # conv6 = BatchNormalization()(conv6)

    conv6 = Activation("relu")(conv6)

    conv7 = Conv2D(int(nch / 2), (3, 3), padding="same", kernel_initializer="glorot_uniform", activation=None,
                   name="Vision_conv7")(conv6)
    # conv7 = BatchNormalization()(conv7)

    conv7 = Activation("relu")(conv7)

    mp3 = MaxPooling2D(pool_size=(2, 2))(conv7)
    drop3 = Dropout(0.5)(mp3)
    # Conv 8 and 9 and 10
    conv8 = Conv2D(nch, (3, 3), padding="same", kernel_initializer="glorot_uniform", activation=None,
                   name="Vision_conv8")(drop3)
    # conv8 = BatchNormalization()(conv8)

    conv8 = Activation("relu")(conv8)

    conv9 = Conv2D(nch, (3, 3), padding="same", kernel_initializer="glorot_uniform", activation=None,
                   name="conv9")(conv8)

    # conv9 = BatchNormalization()(conv9)

    conv9 = Activation("relu")(conv9)

    conv10 = Conv2D(nch, (3, 3), padding="same", kernel_initializer="glorot_uniform", activation=None,
                    name="conv10")(conv9)

    conv10 = Activation("relu")(conv10)

    if smallNetwork:

        if inhibition:

            conv4_inhibition = Conv2D(int(nch / 2), (3, 3), padding="same", kernel_initializer="glorot_uniform",
                                       activation="relu",
                                       name="conv4_Inhibition")(conv3)

            inhibition = ShuntingInhibition(name="inhibitoryLayer", shape=(1,1,int(nch / 2)))([conv4, conv4_inhibition])

            mp4 = MaxPooling2D(pool_size=(2, 2))(inhibition)
            dropSmall = Dropout(0.5)(mp4)

        else:
            mp4 = MaxPooling2D(pool_size=(2, 2))(conv4)
            dropSmall = Dropout(0.5)(mp4)

        flatten = Flatten()(dropSmall)
    else:
        if inhibition:

            conv10_inhibition = Conv2D(nch, (3, 3), padding="same", kernel_initializer="glorot_uniform",
                                      activation=None,
                                      name="conv10_Inhibition")(conv9)

            conv10_inhibition = Activation("relu")(conv10_inhibition)

            inhibition = ShuntingInhibition(name="inhibitoryLayer", shape=(1,1,int(nch)))([conv10, conv10_inhibition])

            mp4 = MaxPooling2D(pool_size=(2, 2))(inhibition)
            dropBig = Dropout(0.5)(mp4)

        else:
            mp4 = MaxPooling2D(pool_size=(2, 2))(conv10)
            dropBig = Dropout(0.5)(mp4)

        flatten = Flatten()(dropBig)

    dense = Dense(denseLayer, activation="relu", name="denseLayer")(flatten)
    drop5 = Dropout(0.5)(dense)

    arousal_output = Dense(units=1, activation="tanh", name='arousal_output')(drop5)

    model = Model(inputs=inputLayer, outputs=arousal_output)


    """Training Parameters"""

    if decay:
        optimizer = SGD(learning_rate=initialLR, momentum=momentum, nesterov=nesterov)
    else:
        optimizer = SGD(lr=initialLR, momentum=momentum, nesterov=nesterov, decay=0.1 / 10)


    model.compile(loss={'arousal_output': 'mean_squared_error'},
                  optimizer=optimizer,
                  metrics=[ccc])


    return model