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

def getClass(value):

    # print ("Value:" + str(value))
    value = float(value)
    absValue = abs(value) / 0.2
    # print ("abs:" + str(absValue))
    newClass = round(absValue,0)
    # print ("newclass:" + str(newClass))
    if value >= 0:
        newClass = 5 + newClass
        # print ("newclass:" + str(newClass))
    else:
        newClass = 5 - newClass

    # print ("Class:" + str(value) + " - new class:" + str(newClass))
    # input ("here")
    return int(newClass)


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

    if len(bags[indexAddToBag]) > 5000:
        addIt = False
    else:
        bags[indexAddToBag].append(value)
        addIt = True

    return addIt, bags
#
# def bagIt(valueA, valueV, bagsA, bagsV):
#
#
#
#
#     addIt = False
#
#     if valueA >= -1.0 and valueA < -0.8:
#         indexAdd = 0
#     elif valueA >= -0.8 and valueA < -0.6:
#         indexAdd = 1
#     elif valueA >= -0.6 and valueA < -0.4:
#         indexAdd = 2
#     elif valueA >= -0.4 and valueA < -0.2:
#         indexAdd = 3
#     elif valueA >= -0.2 and valueA < 0:
#         indexAdd = 4
#
#     elif valueA >= 0 and valueA < 0.2:
#         indexAdd = 5
#
#     elif valueA >= 0.2 and valueA < 0.4:
#         indexAdd = 6
#
#     elif valueA >= 0.4 and valueA < 0.6:
#         indexAdd = 7
#
#     elif valueA >= 0.6 and valueA < 0.8:
#         indexAdd = 8
#
#     elif valueA >= 0.8 and valueA <= 1:
#         indexAdd = 9
#
#
#     if len(bagsA[indexAdd]) > 5000:
#         addIt = False
#     else:
#         bagsA[indexAdd].append(valueA)
#         addIt = True
#
#     #
#     # # if len(bagsA[indexAddToBagA]) > 1000 or len(bagsV[indexAddToBagV]) > 1000:
#     #     addIt = False
#     # else:
#     #     bagsA[indexAddToBagA].append(valueA)
#     #     bagsV[indexAddToBagV].append(valueV)
#     #     addIt = True
#     #
#     # indexAddToBagA = getClass(valueA)
#     # indexAddToBagV = getClass(valueV)
#     #
#     # if len(bagsA[indexAddToBagA]) > 10000:
#     #
#     # # if len(bagsA[indexAddToBagA]) > 1000 or len(bagsV[indexAddToBagV]) > 1000:
#     #     addIt = False
#     # else:
#     #     bagsA[indexAddToBagA].append(valueA)
#     #     bagsV[indexAddToBagV].append(valueV)
#     #     addIt = True
#
#     return addIt, bagsA, bagsV


# import tensorflow as tf
# tf.config.gpu.set_per_process_memory_fraction(0.75)
# tf.config.gpu.set_per_process_memory_growth(True)


#LOCAL
# import tensorflow as tf
# config = tf.ConfigProto()
# config.gpu_options.allow_growth=True
# sess = tf.Session(config=config)

videoDirectory = "/home/pablo/Documents/Datasets/affwild2/cropped_aligned"

trainingLabelDirectory = "/home/pablo/Documents/Datasets/affwild2/annotations-20200917T112933Z-001/annotations/VA_Set/Training_Set"

# trainingLabelDirectory = "/home/pablo/Documents/Datasets/affwild2/balancedAnnotationsTraining/"
validationLabelDirectory = "/home/pablo/Documents/Datasets/affwild2/annotations-20200917T112933Z-001/annotations/VA_Set/Validation_Set"
saveExperiment = "/home/pablo/Documents/Datasets/FaceChannel_Outputs/AffWild2/savedModel"
logFolder = "/home/pablo/Documents/Datasets/FaceChannel_Outputs/AffWild2/log"

expressionLabels = "/home/pablo/Documents/Datasets/affwild2/annotations-20200917T112933Z-001/annotations/EXPR_Set/Training_Set"

savedModel = "/home/pablo/Documents/Workspace/FaceChannel/TrainedNetworks/DimensionalFaceChannel.h5"

useClasses = False

#GCLOUD
# videoDirectory = "/home/pablovin/dataset/affwild2/cropped_aligned"
#
# trainingLabelDirectory = "/home/pablovin/dataset/affwild2/annotations/VA_Set/Training_Set"
# validationLabelDirectory = "/home/pablovin/dataset/affwild2/annotations/VA_Set/Validation_Set"
# saveExperiment = "/home/pablovin/experiments/facechannel/savedModels"
# logFolder = "/home/pablovin/experiments/facechannel/log"
#
# savedModel = "/home/pablovin/experiments/facechannel/weights.01-0.02.h5"


imageSize = (114,114)
inputShape = numpy.array((114, 114, 3)).astype(numpy.int16)

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
            if  float(valence) >= -1 and float(valence) <= 1 and float(arousal) >=-1 and float(arousal) <= 1 :
                fileNumber = str(rowNumber)
                while not len(str(fileNumber)) == 5:
                    fileNumber = "0"+fileNumber

                fileName = videoDirectory + "/" + file.split(".")[0] + "/" + str(fileNumber) + ".jpg"
                if os.path.exists(fileName):
                    validationSamples.append(fileName)
                    if useClasses:
                        newArousal = numpy.zeros(21)
                        newArousal[getClass(arousal)] = 1

                        newValence = numpy.zeros(21)
                        newValence[getClass(valence)] = 1

                        validationLabels.append([newArousal,newValence])

                    else:
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
    if os.path.exists(expressionLabels+"/"+file):
        labelFile = open(trainingLabelDirectory+"/"+file)
        expressionFile = open(expressionLabels+"/"+file)
        expressionLines = expressionFile.readlines()

        rowNumber = 0
        for line in labelFile:
            if rowNumber > 0:
                valence,arousal = line.split(",")
                expression = expressionLines[rowNumber]

                if float(arousal) >= -1 and float(arousal) <= 1:
                    fileNumber = str(rowNumber)
                    while not len(str(fileNumber)) == 5:
                        fileNumber = "0"+fileNumber

                    # Check for affective consistence
                    fileName = videoDirectory + "/" + file.split(".")[0] + "/" + str(fileNumber) + ".jpg"
                    if os.path.exists(fileName):
                        trainingSamples.append(fileName)
                        if useClasses:
                            newArousal = numpy.zeros(21)
                            newArousal[getClass(arousal)] = 1

                            newValence = numpy.zeros(21)
                            newValence[getClass(valence)] = 1

                            trainingLabels.append([newArousal, newValence])

                        else:
                            trainingLabels.append([float(arousal), float(valence)])


                        arousals.append(float(arousal))
                        valences.append(float(valence))

            rowNumber +=1

trainingSamples,trainingLabels = shuffleData(trainingSamples, trainingLabels)
validationSamples,validationLabels = shuffleData(validationSamples, validationLabels)


bagsA = []
bagsV = []
for a in range(10):
    bagsV.append([])
    bagsA.append([])

# normalizedTrainingSamples, normalizedTrainingLabels = trainingSamples[0:30000],trainingLabels[0:30000]


normalizedTrainingSamples, normalizedTrainingLabels = [],[]
for a in range(len(trainingLabels)):

    arousal,valence = trainingLabels[a]
    add, bagsA = bagIt(arousal, bagsA)
    if add:
        normalizedTrainingSamples.append(trainingSamples[a])
        normalizedTrainingLabels.append([arousal, valence])
#
# for a in range(10):
#     print ("Bags:" + str(len(bags[a])))

print ("Bags Arousal")
for a in range(10):
    print ("Bags:" + str(len(bagsA[a])))

# print ("Bags Valence")
# for a in range(10):
#     print ("Bags:" + str(len(bagsV[a])))

normalizedTrainingSamples = numpy.array(normalizedTrainingSamples)[0:30000]
normalizedTrainingLabels = numpy.array(normalizedTrainingLabels)[0:30000]

print ("Samples:" + str(normalizedTrainingSamples.shape))
print ("Labels:" + str(normalizedTrainingLabels.shape))
# print ("Validation:" + str(len(validationLabels)))


createHistogram(normalizedTrainingLabels[:, 0], normalizedTrainingLabels[:, 1], "/home/pablo/Documents/Datasets/FaceChannel_Outputs/AffWild2/dataVis", "training_all")

# input("here")

# print ("Shape samples:" + str(normalizedTrainingSamples.shape))
#
# print ("Shape labels:" + str(normalizedTrainingLabels.shape))

trainingSamples = normalizedTrainingSamples
trainingLabels = normalizedTrainingLabels

#
# trainingSamples = normalizedTrainingSamples
# trainingLabels = normalizedTrainingLabels

newValidationSamples = validationSamples[0:1000]
newValidationLabels = validationLabels[0:1000]

trainingLabels = numpy.array(trainingLabels)
validationLabels = numpy.array(validationLabels)

createHistogram(arousals, valences, "/home/pablo/Documents/Datasets/FaceChannel_Outputs/AffWild2/dataVis", "training_all")

createHistogram(normalizedTrainingLabels[:,0], normalizedTrainingLabels[:, 1], "/home/pablo/Documents/Datasets/FaceChannel_Outputs/AffWild2/dataVis", "training_normalized")
createHistogram(trainingLabels[:,0], trainingLabels[:, 1], "/home/pablo/Documents/Datasets/FaceChannel_Outputs/AffWild2/dataVis", "trainnig_trainModel")
createHistogram(validationLabels[:,0], validationLabels[:, 1], "/home/pablo/Documents/Datasets/FaceChannel_Outputs/AffWild2/dataVis", "validation_trainModel")
# print ("Len arousals:" + str(len(arousals)))
# input("here")
print ("Training:" + str(len(trainingLabels)))
print ("Testing:" + str(len(testingSamples)))
print ("Validation:" + str(len(validationLabels)))
# input("here")
model = Model.buildModel(inputShape, 8)
# model = Model.buildVgg16()

from metrics import fbeta_score, recall, precision, rmse, ccc
# model = Model.loadModel(saveExperiment)

# Model.evaluate(model,  [validationSamples, validationLabels], imageSize)

model = Model.train(model, [trainingSamples, trainingLabels], [testingSamples,testingLabels], [newValidationSamples, newValidationLabels], imageSize, saveExperiment, logFolder)

Model.evaluate(model,  [validationSamples, validationLabels], imageSize)
