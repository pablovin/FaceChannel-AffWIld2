import csv
import numpy

import Model

import cv2
import random

import os

from dataVisualization import createHistogram



from imgaug import augmenters as iaa
import imgaug as ia

st = lambda aug: iaa.Sometimes(0.5, aug)

seq = iaa.Sequential([
    iaa.Fliplr(0.5),
    # st(iaa.Add((-10, 10), per_channel=0.5)),
    # st(iaa.Multiply((0.5, 1.5), per_channel=0.5)),
    # st(iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5)),
    st(iaa.Affine(
        scale={"x": (0.9, 1.10), "y": (0.9, 1.10)},
        # scale images to 80-120% of their size, individually per axis
        translate_px={"x": (-5, 5), "y": (-5, 5)},  # translate by -16 to +16 pixels (per axis)
        rotate=(-10, 10),  # rotate by -45 to +45 degrees
        shear=(-3, 3),  # shear by -16 to +16 degrees
        order=3,  # use any of scikit-image's interpolation methods
        cval=(0.0, 1.0),  # if mode is constant, use a cval between 0 and 1.0
        mode="constant"
    )),
], random_order=True)


def shuffleData(dataX, dataY):
    positions = []
    for p in range(len(dataX)):
        positions.append(p)

    random.shuffle(positions)

    newInputs = []
    newOutputs = []
    for p in positions:
        newInputs.append(dataX[p])
        newOutputs.append(dataY[p])

    return (newInputs, newOutputs)


def preProcess(dataLocation, imageSize, grayScale):

    # print ("Location:" + str(dataLocation))
    frame = cv2.imread(dataLocation)

    data = numpy.array(cv2.resize(frame, imageSize))

    if grayScale:
       data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
       data = numpy.expand_dims(data, axis=0)


    # data = numpy.swapaxes(data, 1, 2)
    # data = numpy.swapaxes(data, 0, 1)

    data = data.astype('float32')

    data /= 255

    data = numpy.array(data)

    return data


def bagIt(value, bags):

    addIt = False
    indexAddToBag = 0

    if value < -1 and value > -0.8:
        indexAddToBag = 0
    elif value < -0.8 and value > -0.6:
        indexAddToBag = 1
    elif value < -0.6 and value > -0.4:
        indexAddToBag = 2
    elif value < - 0.4 and value > -0.2:
        indexAddToBag = 3
    elif value < - 0.2 and value < 0.0:
        indexAddToBag = 4

    if value < 1 and value > 0.8:
        indexAddToBag = 5
    elif value < 0.8 and value > 0.6:
        indexAddToBag = 6
    elif value < 0.6 and value > 0.4:
        indexAddToBag = 7
    elif value < 0.4 and value > 0.2:
        indexAddToBag = 8
    elif value < 0.2 and value > 0.0:
        indexAddToBag = 9

    if len(bags[indexAddToBag]) > 100000:
        addIt = False
    else:
        bags[indexAddToBag].append(value)
        addIt = True

    return addIt, bags
#LOCAL
# import tensorflow as tf
# config = tf.ConfigProto()
# config.gpu_options.allow_growth=True
# sess = tf.Session(config=config)
#
# videoDirectory = "/home/pablo/Documents/Datasets/affwild2/cropped_aligned"
#
# trainingLabelDirectory = "/home/pablo/Documents/Datasets/affwild2/annotations-20200917T112933Z-001/annotations/VA_Set/Training_Set"
# validationLabelDirectory = "/home/pablo/Documents/Datasets/affwild2/annotations-20200917T112933Z-001/annotations/VA_Set/Validation_Set"
# saveExperiment = "/home/pablo/Documents/Datasets/FaceChannel_Outputs/AffWild2/savedModel"
# logFolder = "/home/pablo/Documents/Datasets/FaceChannel_Outputs/AffWild2/log"
#
#
# savedModel = "/home/pablo/Documents/Workspace/FaceChannel/TrainedNetworks/DimensionalFaceChannel.h5"


#GCLOUD
videoDirectory = "/home/pablovin/dataset/affwild2/cropped_aligned"

trainingLabelDirectory = "/home/pablovin/dataset/affwild2/annotations/VA_Set/Training_Set"
validationLabelDirectory = "/home/pablovin/dataset/affwild2/annotations/VA_Set/Validation_Set"
saveExperiment = "/home/pablovin/experiments/facechannel/savedModels"
logFolder = "/home/pablovin/experiments/facechannel/log"

savedModel = "/home/pablovin/experiments/facechannel/weights.01-0.02.h5"


imageSize = (112,112)
inputShape = numpy.array((112, 112, 3)).astype(numpy.int32)

trainingSamples = []
trainingLabels = []

testingSamples = []
testingLabels = []

validationSamples = []
validationLabels = []

arousals = []
valences = []
print ("---- Reading validation")
for file in os.listdir(validationLabelDirectory):
    print ("Reading file:" + str(file))
    labelFile = open(validationLabelDirectory+"/"+file)
    rowNumber = 0
    for line in labelFile:
        if rowNumber > 0:
            valence,arousal = line.split(",")
            if  float(valence) >= -1 and float(valence) <= 1:
                fileNumber = str(rowNumber)
                while not len(str(fileNumber)) == 5:
                    fileNumber = "0"+fileNumber

                fileName = videoDirectory + "/" + file.split(".")[0] + "/" + str(fileNumber) + ".jpg"
                if os.path.exists(fileName):
                    validationSamples.append(fileName)
                    validationLabels.append([float(arousal),float(valence)])
                    arousals.append(float(arousal))
                    valences.append(float(valence))

        rowNumber +=1

# print ("Len arousals:" + str(len(arousals)))
# createHistogram(arousals, valences, "/home/pablo/Documents/Datasets/FaceChannel_Outputs/AffWild2/dataVis", "validation")

print ("---- Reading training")
arousals = []
valences = []



for file in os.listdir(trainingLabelDirectory):
    print ("Reading file:" + str(file))
    labelFile = open(trainingLabelDirectory+"/"+file)
    rowNumber = 0
    for line in labelFile:
        if rowNumber > 0:
            valence,arousal = line.split(",")
            if float(valence) >= -1 and float(valence) <= 1:
                fileNumber = str(rowNumber)
                while not len(str(fileNumber)) == 5:
                    fileNumber = "0"+fileNumber

                fileName = videoDirectory + "/" + file.split(".")[0] + "/" + str(fileNumber) + ".jpg"
                if os.path.exists(fileName):
                    trainingSamples.append(fileName)
                    trainingLabels.append([float(arousal),float(valence)])
                    arousals.append(float(arousal))
                    valences.append(float(valence))

        rowNumber +=1

trainingSamples,trainingLabels = shuffleData(trainingSamples, trainingLabels)
validationSamples,validationLabels = shuffleData(validationSamples, validationLabels)



bags = []
for a in range(10):
    bags.append([])

normalizedTrainingSamples, normalizedTrainingLabels = [],[]

for a in range(len(trainingLabels)):

    arousal,valence = trainingLabels[a]
    add, bags = bagIt(arousal, bags)
    if add:
        normalizedTrainingSamples.append(trainingSamples[a])
        normalizedTrainingLabels.append([arousal, valence])

for a in range(10):
    print ("Bags:" + str(len(bags[a])))


normalizedTrainingSamples = numpy.array(normalizedTrainingSamples)
normalizedTrainingLabels = numpy.array(normalizedTrainingLabels)

# print ("Shape samples:" + str(normalizedTrainingSamples.shape))
#
# print ("Shape labels:" + str(normalizedTrainingLabels.shape))

trainingSamples = normalizedTrainingSamples
trainingLabels = normalizedTrainingLabels


trainingSamples = normalizedTrainingSamples
trainingLabels = normalizedTrainingLabels


validationSamples = validationSamples[0:1000]
validationLabels = validationLabels[0:1000]

trainingLabels = numpy.array(trainingLabels)
validationLabels = numpy.array(validationLabels)

# createHistogram(arousals, valences, "/home/pablo/Documents/Datasets/FaceChannel_Outputs/AffWild2/dataVis", "training_all")
#
# createHistogram(normalizedTrainingLabels[:,0], normalizedTrainingLabels[:, 1], "/home/pablo/Documents/Datasets/FaceChannel_Outputs/AffWild2/dataVis", "training_normalized")
# createHistogram(trainingLabels[:,0], trainingLabels[:, 1], "/home/pablo/Documents/Datasets/FaceChannel_Outputs/AffWild2/dataVis", "training_10k")
# createHistogram(validationLabels[:,0], validationLabels[:, 1], "/home/pablo/Documents/Datasets/FaceChannel_Outputs/AffWild2/dataVis", "validation_10k")
# print ("Len arousals:" + str(len(arousals)))
# input("here")
print ("Training:" + str(len(trainingLabels)))
print ("Testing:" + str(len(testingSamples)))
print ("Validation:" + str(len(validationLabels)))

model = Model.buildModel(inputShape, 8)

# model = Model.loadModel(savedModel)

# Model.evaluate(model,  [validationSamples, validationLabels], imageSize)

Model.train(model, [trainingSamples, trainingLabels], [testingSamples,testingLabels], [validationSamples, validationLabels], imageSize, saveExperiment, logFolder)

Model.evaluate(model,  [validationSamples, validationLabels], imageSize)
