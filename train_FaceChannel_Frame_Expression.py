from DataLoaders import AffWildDataLoader
from Models import EXPClassifier
from Generators import EXPGenerator
from datetime import datetime
import sklearn
import numpy

"""
Experiment Information
"""

#Data location
videoDirectory = "/home/pablo/Documents/Datasets/affwild2/cropped_aligned"
trainingLabelDirectory = "/home/pablo/Documents/Datasets/affwild2/balancedAnnotationsTraining_Exp/"
validationLabelDirectory = "/home/pablo/Documents/Datasets/affwild2/annotations-20200917T112933Z-001/annotations/EXPR_Set/Validation_Set"
experimentFolder = "/home/pablo/Documents/Datasets/FaceChannel_Outputs/AffWild2/Experiments/AffWIld2_Final/Models/Expression_Frame"

timeNow = str(datetime.now())
experimentFolder = experimentFolder + "/" + timeNow

#Type of data, model and generator
dataType = AffWildDataLoader.DATATYPE["Expression_Frame"]
modelType =EXPClassifier.MODELTYPE["Exp_Frame_LoadedModel"]
generatorType =EXPGenerator.GENERATORTYPE["EXP_FaceChannel"]


#Training Parameters
epoches = 10
batchSize = 1024
maxSamplesTraining = 200000
maxSamplesValidation = 1000
trainingBins = 7

shuffle = True

#Image parameters
inputShape = (112,112,3) # Image H, W, RGB


"""
Load data
"""
trainingSamples, trainingLabels = AffWildDataLoader.getData(videoDirectory,trainingLabelDirectory, dataType, shuffle=shuffle, maxSamples=maxSamplesTraining, splitName="Training", loadingBins=trainingBins, histogramDirectory=experimentFolder)

validationSamples, validationLabels = AffWildDataLoader.getData(videoDirectory,validationLabelDirectory, dataType, shuffle=shuffle, maxSamples=maxSamplesValidation, splitName="Validation", histogramDirectory=experimentFolder)

fullValidationSamples, fullValidationLabels = AffWildDataLoader.getData(videoDirectory,validationLabelDirectory, dataType, shuffle=False, splitName="Full_Validation")

# input("here")
"""
Create Generator
"""

trainGenerator = EXPGenerator.getGenerator(generatorType, trainingSamples, trainingLabels, batchSize, inputShape, dataAugmentation=True)
validationGenerator = EXPGenerator.getGenerator(generatorType, validationSamples, validationLabels, batchSize, inputShape)

fullValidationGenerator = EXPGenerator.getGenerator(generatorType, fullValidationSamples, fullValidationLabels, batchSize, inputShape)


"""
Create Model
"""

model = EXPClassifier.getModel(inputShape, modelType)
#
# model = EXPClassifier.loadModel("/home/pablo/Documents/Datasets/FaceChannel_Outputs/AffWild2/Experiments/AffWIld2_Final/Models/Arousal_Frame/2020-10-02 18:20:18.755732/Model")

"""
Train Model
"""
model = EXPClassifier.train(model,experimentFolder,trainGenerator, validationGenerator, batchSize, epoches)

"""
Evaluate Model
"""

print ("Predicting...")
predictions = EXPClassifier.predict(model,fullValidationGenerator)

print ("Indexing...")

indicesP = numpy.argmax(predictions, axis=1)
indicesT = numpy.argmax(fullValidationLabels, axis=1)

print ("Calculating...")
print(sklearn.metrics.classification_report(indicesT, indicesP))
#
# EXPClassifier.evaluate(model, fullValidationGenerator, batchSize)