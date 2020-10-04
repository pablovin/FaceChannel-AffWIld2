from tensorflow import keras
from Metrics.metrics import ccc
from keras.models import Sequential, Input
from keras.layers import Dense, Dropout, Flatten, Lambda, TimeDistributed, LSTM
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.optimizers import Adam, Adamax, SGD
from keras.layers.core import Activation
from keras.models import Model

import tensorflow as tf

from Models.CPC import loadModel as CPCLoad

from Models.Layer_ShuntingInhibition import ShuntingInhibition

import os
from datetime import datetime

MODELTYPE = {'Arousal_Frame_FaceChannel': 'Arousal_Frame_FaceChannel',
             'Arousal_Frame_ResNet50v2': 'Arousal_Frame_ResNet50v2',

             'Arousal_Frame_CPC': 'Arousal_Frame_CPC',


             'Arousal_Frame_FaceChannel_Optmizer': 'Arousal_Frame_FaceChannel_Optmizer',
             'Arousal_Sequence_FaceChannel_Optmizer': 'Arousal_Sequence_FaceChannel_Optmizer',
             "Arousal_Frame_CPC_Optmizer":"Arousal_Frame_CPC_Optmizer",

             'AVExp_Frame_FaceChannel': 'AVExp_Frame_FaceChannel',
             'AVExp_Sequence_FaceChannel': 'AVExp_Sequence_FaceChannel',

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

    elif modelType == MODELTYPE["AVExp_Frame_FaceChannel"]:
        return avEXPFrameFaceChannelModel(inputShape)


    elif modelType == MODELTYPE["Arousal_Sequence_FaceChannel"]:
        return arousalSequenceFaceChannelModel(inputShape)

    elif modelType == MODELTYPE["AVExp_Sequence_FaceChannel"]:
        return avEXPSequenceFaceChannelModel(inputShape)


    elif modelType == MODELTYPE["Arousal_Frame_CPC"]:
        return arousalFrameCPC(inputShape)

    elif modelType == MODELTYPE["Arousal_Frame_CPC_Optmizer"]:
        return arousalFrameCPC_Optmizer(params)

    elif modelType == MODELTYPE["Arousal_Frame_FaceChannel_Optmizer"]:
        return arousalFrameFaceChannelModel_Optmizer(inputShape, params)

    elif modelType == MODELTYPE["Arousal_Sequence_FaceChannel_Optmizer"]:
        return arousalSequenceFaceChannelModel_Optmizer(inputShape, params)


"""
Utils
"""

def predict(model, testGenerator):

    predictions = model.predict(testGenerator)

    return predictions


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
        # lstmSize, denseSize, batchSize = params

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

def arousalSequenceFaceChannelModel_Optmizer(inputShape, params):
    # keras.backend.set_image_data_format("channels_first")

    lstmSize, denseSize, batchSize = params

    modelLoaded = keras.models.load_model("/home/pablo/Documents/Datasets/FaceChannel_Outputs/AffWild2/Optmization/Round2_60kImages/2020-10-01 05:05:18.686795/Model",
                                    custom_objects={'ccc': ccc})

    modelDense =  modelLoaded.get_layer(name="denseLayer")

    previousModel = Model(inputs=modelLoaded.input, outputs=modelDense.output)

    input_layer = Input(shape=inputShape)

    # model = Sequential()

    # Conv1 and 2

    td1 = TimeDistributed(previousModel, name="FaceChannel")(input_layer)

    # Flatten
    flatten = TimeDistributed(Flatten(), name="Flatten") (td1)

    # RNN

    lstm = LSTM(lstmSize, activation='relu', return_sequences=False, name="Rnn_1") (flatten)

    dense = Dense(denseSize, activation='relu', name="dense_1") (lstm)

    drop1 = Dropout(0.5, name="drop5")(dense)

    arousal_output = Dense(units=1, activation='tanh', name='arousal_output')(drop1)

    model = Model(inputs=input_layer, outputs=arousal_output)

    for layer in model.layers:
        layer.trainable = False

    model.get_layer(name="Rnn_1").trainable = True
    model.get_layer(name="dense_1").trainable = True
    model.get_layer(name="arousal_output").trainable = True


    optimizer = Adam()

    model.compile(loss={'arousal_output': 'mean_squared_error'},
                  optimizer=optimizer,
                  metrics=[ccc])

    return model


def arousalSequenceFaceChannelModel(inputShape):
    modelLoaded = keras.models.load_model(
        "/home/pablo/Documents/Datasets/FaceChannel_Outputs/AffWild2/Optmization/Round2_60kImages/2020-10-01 05:05:18.686795/Model",
        custom_objects={'ccc': ccc})

    modelDense = modelLoaded.get_layer(name="denseLayer")

    previousModel = Model(inputs=modelLoaded.input, outputs=modelDense.output)

    input_layer = Input(shape=inputShape)

    # model = Sequential()

    # Conv1 and 2

    td1 = TimeDistributed(previousModel, name="FaceChannel")(input_layer)

    # Flatten
    flatten = TimeDistributed(Flatten(), name="Flatten")(td1)

    # RNN

    lstm = LSTM(100, activation='relu', return_sequences=False, name="Rnn_1")(flatten)

    dense = Dense(100, activation='relu', name="dense_1")(lstm)

    drop1 = Dropout(0.5, name="drop5")(dense)

    arousal_output = Dense(units=1, activation='tanh', name='arousal_output')(drop1)

    model = Model(inputs=input_layer, outputs=arousal_output)

    # for layer in model.layers:
    #     layer.trainable = False
    #
    # model.get_layer(name="Rnn_1").trainable = True
    # model.get_layer(name="dense_1").trainable = True
    # model.get_layer(name="arousal_output").trainable = True

    optimizer = SGD(learning_rate= 0.015, decay=0.1 / 10, momentum=0.74, nesterov=True)
    # optimizer = SGD(lr=0.1, momentum=0.9, decay=0.1 / 10)

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


def arousalFrameCPC(inputShape):


    # new_input = Input(shape=inputShape)

    crop_shape = (16, 16, 3)
    n_crops = 7
    code_size = 128

    encoder = CPCLoad(
        "/home/pablo/Documents/Datasets/FaceChannel_Outputs/AffWild2/Experiments/CPC/200kI_Images_2020-10-01 21:06:14.473694/Model/Encoder")

    # encoder.summary()
    # input("here")
    # encoder.trainable = False
    # for l in encoder.layers:
    #     l.trainable = False

    # Crops feature extraction
    x_input = keras.layers.Input((n_crops, n_crops) + crop_shape)
    x = keras.layers.Reshape((n_crops * n_crops, ) + crop_shape)(x_input)
    x = keras.layers.TimeDistributed(encoder)(x)
    x = keras.layers.Reshape((n_crops, n_crops, code_size))(x)

    # previousModel = Model(inputs=modelLoaded.input, outputs=modelDense.output)

    # x = Flatten(x)

    x = keras.layers.GlobalAveragePooling2D()(x)

    dense = Dense(500, activation="relu", name="denseLayer")(x)
    drop5 = Dropout(0.5)(dense)

    arousal_output = Dense(units=1, activation='tanh', name='arousal_output')(drop5)

    model = Model(inputs=x_input, outputs=arousal_output)

    # for layer in model.layers:
    #     layer.trainable = False
    #
    # model.get_layer(name="denseLayer").trainable = True
    # model.get_layer(name="arousal_output").trainable = True

    optimizer = Adam()
    # optimizer = SGD(learning_rate=0.015, decay=0.1 / 10, momentum=0.74, nesterov=True)
    # optimizer = SGD(lr=0.1, momentum=0.9, decay=0.1 / 10)

    model.compile(loss={'arousal_output': 'mean_squared_error'},
                  optimizer=optimizer,
                  metrics=[ccc])


    return model



def arousalFrameCPC_Optmizer(params):


    denseSize, optmizer, initialLearningRate, momentum = params
    # new_input = Input(shape=inputShape)

    crop_shape = (16, 16, 3)
    n_crops = 7
    code_size = 128

    encoder = CPCLoad(
        "/home/pablo/Documents/Datasets/FaceChannel_Outputs/AffWild2/Experiments/CPC/200kI_Images_2020-10-01 21:06:14.473694/Model/Encoder")

    # encoder.summary()
    # input("here")
    # encoder.trainable = False
    # for l in encoder.layers:
    #     l.trainable = False

    # Crops feature extraction
    x_input = keras.layers.Input((n_crops, n_crops) + crop_shape)
    x = keras.layers.Reshape((n_crops * n_crops, ) + crop_shape)(x_input)
    x = keras.layers.TimeDistributed(encoder)(x)
    x = keras.layers.Reshape((n_crops, n_crops, code_size))(x)

    # previousModel = Model(inputs=modelLoaded.input, outputs=modelDense.output)

    # x = Flatten(x)

    x = keras.layers.GlobalAveragePooling2D()(x)

    dense = Dense(denseSize, activation="relu", name="denseLayer")(x)
    drop5 = Dropout(0.5)(dense)

    arousal_output = Dense(units=1, activation='tanh', name='arousal_output')(drop5)

    model = Model(inputs=x_input, outputs=arousal_output)

    # for layer in model.layers:
    #     layer.trainable = False
    #
    # model.get_layer(name="denseLayer").trainable = True
    # model.get_layer(name="arousal_output").trainable = True

    if optmizer == "Adam":
        optimizer = Adam()
    else:
        optimizer = SGD(learning_rate=initialLearningRate, decay=0.1 / 10, momentum=momentum, nesterov=True)
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

    mp4 = MaxPooling2D(pool_size=(2, 2))(conv10)
    drop4 = Dropout(0.5)(mp4)


    #
    flatten = Flatten()(drop4)

    dense = Dense(500, activation="relu", name="denseLayer")(flatten)
    drop5 = Dropout(0.5)(dense)

    arousal_output = Dense(units=1, activation='tanh', name='arousal_output')(drop5)

    model = Model(inputs=inputLayer, outputs=arousal_output)


    """Training Parameters"""

    optimizer = SGD(learning_rate= 0.015, decay=0.1 / 10, momentum=0.74, nesterov=True)
    # optimizer = SGD(lr=0.1, momentum=0.9, decay=0.1 / 10)

    model.compile(loss={'arousal_output': 'mean_squared_error'},
                  optimizer=optimizer,
                  metrics=[ccc])


    return model



def avEXPSequenceFaceChannelModel(inputShape):


    modelLoaded = keras.models.load_model(
        "/home/pablo/Documents/Datasets/FaceChannel_Outputs/AffWild2/Experiments/AffWIld2_Final/Models/AV_Exp/50k_BestAcc_2020-10-03 20:45:28.043769/Model",
        custom_objects={'ccc': ccc})

    modelDense = modelLoaded.get_layer(name="denseLayer")

    previousModel = Model(inputs=modelLoaded.input, outputs=modelDense.output)

    # encoder.summary()
    # input("here")
    # previousModel.trainable = False
    # for l in previousModel.layers:
    #     l.trainable = False

    input_layer = Input(shape=inputShape)

    td1 = TimeDistributed(previousModel, name="FaceChannel")(input_layer)

    # Flatten
    flatten = TimeDistributed(Flatten(), name="Flatten")(td1)

    # RNN

    lstmA = LSTM(100, activation='relu', return_sequences=False, name="Rnn_A")(flatten)
    denseA = Dense(100, activation='relu', name="dense_1")(lstmA)
    drop4 = Dropout(0.5)(denseA)
    arousal_output = Dense(units=1, activation='tanh', name='arousal_output')(drop4)


    lstmV = LSTM(100, activation='relu', return_sequences=False, name="Rnn_V")(flatten)
    denseV = Dense(100, activation="relu", name="denseLayer_V")(lstmV)
    drop7 = Dropout(0.5)(denseV)
    valence_output = Dense(units=1, activation='tanh', name='valence_output')(drop7)

    lstmE = LSTM(100, activation='relu', return_sequences=False, name="Rnn_E")(flatten)
    denseE = Dense(100, activation="relu", name="denseLayer_E")(lstmE)
    drop8 = Dropout(0.5)(denseE)
    exp_output = Dense(units=7, activation='softmax', name='exp_output1')(drop8)

    model = Model(inputs=input_layer, outputs=[arousal_output, valence_output, exp_output])


    """Training Parameters"""

    # optimizer = SGD(learning_rate= 0.015, decay=0.1 / 10, momentum=0.74, nesterov=True)
    optimizer = Adam()
    # optimizer = SGD(lr=0.1, momentum=0.9, decay=0.1 / 10)

    model.compile(loss={'arousal_output': 'mean_squared_error',
                        'valence_output': 'mean_squared_error',
                        'exp_output1': tf.keras.losses.CategoricalCrossentropy()

                        },
                  optimizer=optimizer,
                  metrics = {'arousal_output': ccc,
                        'valence_output': ccc,
                        'exp_output1': "accuracy"

                        })

    # model.compile(loss={'arousal_output': 'mean_squared_error',
    #                     'exp_output': tf.keras.losses.CategoricalCrossentropy()},
    #               optimizer=optimizer)


    return model



def avEXPFrameFaceChannelModel(inputShape):


    modelLoaded = keras.models.load_model(
        "/home/pablo/Documents/Datasets/FaceChannel_Outputs/AffWild2/Experiments/AffWIld2_Final/Models/Arousal_Frame/2020-10-02 18:20:18.755732/Model",
        custom_objects={'ccc': ccc})

    modelDense = modelLoaded.get_layer(name="denseLayer")

    previousModel = Model(inputs=modelLoaded.input, outputs=modelDense.output)

    # encoder.summary()
    # input("here")
    # previousModel.trainable = False
    # for l in previousModel.layers:
    #     l.trainable = False

    #
    flatten = Flatten()(previousModel.output)

    denseA = Dense(100, activation="relu", name="denseLayer_A")(flatten)
    drop6 = Dropout(0.5)(denseA)

    arousal_output = Dense(units=1, activation='tanh', name='arousal_output')(drop6)

    denseV = Dense(100, activation="relu", name="denseLayer_V")(flatten)
    drop7 = Dropout(0.5)(denseV)

    valence_output = Dense(units=1, activation='tanh', name='valence_output')(drop7)

    denseE = Dense(100, activation="relu", name="denseLayer_E")(flatten)
    drop8 = Dropout(0.5)(denseE)
    exp_output = Dense(units=7, activation='softmax', name='exp_output1')(drop8)

    model = Model(inputs=previousModel.input, outputs=[arousal_output, valence_output, exp_output])

    # model.summary()
    # input("here")
    """Training Parameters"""

    # optimizer = SGD(learning_rate= 0.015, decay=0.1 / 10, momentum=0.74, nesterov=True)
    optimizer = Adam()
    # optimizer = SGD(lr=0.1, momentum=0.9, decay=0.1 / 10)

    model.compile(loss={'arousal_output': 'mean_squared_error',
                        'valence_output': 'mean_squared_error',
                        'exp_output1': tf.keras.losses.CategoricalCrossentropy()

                        },
                  optimizer=optimizer,
                  metrics = {'arousal_output': ccc,
                        'valence_output': ccc,
                        'exp_output1': "accuracy"

                        })

    # model.compile(loss={'arousal_output': 'mean_squared_error',
    #                     'exp_output': tf.keras.losses.CategoricalCrossentropy()},
    #               optimizer=optimizer)


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