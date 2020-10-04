from DataLoaders import AffWildDataLoader
from Models import AVClassifier
from Generators import AVGenerator
from datetime import datetime

"""
Experiment Information
"""

#Data location
videoDirectory = "/home/pablo/Documents/Datasets/affwild2/cropped_aligned"
trainingLabelDirectory = "/home/pablo/Documents/Datasets/affwild2/balancedAnnotationsTraining/"
validationLabelDirectory = "/home/pablo/Documents/Datasets/affwild2/annotations-20200917T112933Z-001/annotations/VA_Set/Validation_Set"
experimentFolder = "/home/pablo/Documents/Datasets/FaceChannel_Outputs/AffWild2/Experiments/AffWIld2_Final/Arousal_Sequence"


# #Gcloud
# videoDirectory = "/home/pablovin/dataset/affwild2/cropped_aligned"
# trainingLabelDirectory = "/home/pablovin/dataset/affwild2/balancedAnnotationsTraining"
# validationLabelDirectory = "/home/pablovin/dataset/affwild2/annotations/VA_Set/Validation_Set"
# experimentFolder = "/home/pablovin/experiments/facechannel/sequence"
#


timeNow = str(datetime.now())
experimentFolder = experimentFolder + "/" + timeNow

#Type of data, model and generator
dataType = AffWildDataLoader.DATATYPE["Arousal_Sequence"]
modelType =AVClassifier.MODELTYPE["Arousal_Sequence_FaceChannel"]
generatorType =AVGenerator.GENERATORTYPE["Arousal_FaceChannel"]

#Training Parameters
epoches = 10
batchSize = 128
maxSamplesTraining = 30000
maxSamplesValidation = 1000
trainingBins = 21
sequenceSize = 10

shuffle = True

#Image parameters
inputShape = (sequenceSize, 112,112,3) # Image H, W, RGB

"""
Load data
"""
trainingSamples, trainingLabels = AffWildDataLoader.getData(videoDirectory,trainingLabelDirectory, dataType, shuffle=shuffle, maxSamples=maxSamplesTraining, splitName="Training", loadingBins=trainingBins, histogramDirectory=experimentFolder, sequenceSize=sequenceSize)

validationSamples, validationLabels = AffWildDataLoader.getData(videoDirectory,validationLabelDirectory, dataType, shuffle=shuffle, maxSamples=maxSamplesValidation, splitName="Validation", histogramDirectory=experimentFolder, sequenceSize=sequenceSize)

fullValidationSamples, fullValidationLabels = AffWildDataLoader.getData(videoDirectory,validationLabelDirectory, dataType, shuffle=shuffle, splitName="Full_Validation", sequenceSize=sequenceSize)

# input("here")
"""
Create Generator
"""

trainGenerator = AVGenerator.getGenerator(generatorType, trainingSamples, trainingLabels, batchSize, inputShape, dataAugmentation=False, sequence=True)
validationGenerator = AVGenerator.getGenerator(generatorType, validationSamples, validationLabels, batchSize, inputShape, sequence=True)

fullValidationGenerator = AVGenerator.getGenerator(generatorType, fullValidationSamples, fullValidationLabels, batchSize, inputShape, sequence=True)


"""
Create Model
"""

model = AVClassifier.getModel(inputShape, modelType)

# model = AVClassifier.loadModel("/home/pablo/Documents/Datasets/FaceChannel_Outputs/AffWild2/Experiments/AffWIld2_Final/Arousal_Frame/2020-10-02 18:20:18.755732/Model")

"""
Train Model
"""
model = AVClassifier.train(model,experimentFolder,trainGenerator, validationGenerator, batchSize, epoches)

"""
Evaluate Model
"""

AVClassifier.evaluate(model, fullValidationGenerator, batchSize)