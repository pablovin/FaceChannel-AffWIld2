from DataLoaders import AffWildDataLoader
from Models import AVClassifier
from Generators import AVGenerator
from datetime import datetime

"""
Experiment Information
"""

#Data location
videoDirectory = "/home/pablo/Documents/Datasets/affwild2/cropped_aligned"
trainingLabelAVDirectory = "/home/pablo/Documents/Datasets/affwild2/balancedAnnotationsTraining"
trainingLabelExpDirectory = "/home/pablo/Documents/Datasets/affwild2/annotations-20200917T112933Z-001/annotations/EXPR_Set/Training_Set"

validationAVLabelDirectory = "/home/pablo/Documents/Datasets/affwild2/annotations-20200917T112933Z-001/annotations/VA_Set/Validation_Set"
validationEXPLabelDirectory = "/home/pablo/Documents/Datasets/affwild2/annotations-20200917T112933Z-001/annotations/EXPR_Set/Validation_Set"

experimentFolder = "/home/pablo/Documents/Datasets/FaceChannel_Outputs/AffWild2/Experiments/AffWIld2_Final/Models/AV_Exp"

timeNow = str(datetime.now())
experimentFolder = experimentFolder + "/" + timeNow

#Type of data, model and generator
dataType = AffWildDataLoader.DATATYPE["AV_Exp_Sequence"]
modelType =AVClassifier.MODELTYPE["AVExp_Sequence_FaceChannel"]
generatorType =AVGenerator.GENERATORTYPE["AVExp_FaceChannel"]


#Training Parameters
epoches = 5
batchSize = 128
maxSamplesTraining = 10000
maxSamplesValidation = 1000
trainingBins = 7
sequenceSize = 10

shuffle = True

#Image parameters
inputShape = (sequenceSize, 112,112,3) # Image H, W, RGB

trainingLabelDirectory = [trainingLabelAVDirectory, trainingLabelExpDirectory]
validationLabelDirectory = [validationAVLabelDirectory, validationEXPLabelDirectory]

"""
Load data
"""
trainingSamples, trainingLabels = AffWildDataLoader.getData(videoDirectory,trainingLabelDirectory, dataType, shuffle=shuffle, maxSamples=maxSamplesTraining, loadingBins=trainingBins, splitName="Training",  histogramDirectory=experimentFolder)

validationSamples, validationLabels = AffWildDataLoader.getData(videoDirectory,validationLabelDirectory, dataType, shuffle=shuffle, maxSamples=maxSamplesValidation, splitName="Validation", histogramDirectory=experimentFolder)

fullValidationSamples, fullValidationLabels = AffWildDataLoader.getData(videoDirectory,validationLabelDirectory, dataType, shuffle=shuffle, splitName="Full_Validation")

# input("here")
"""
Create Generator
"""

trainGenerator = AVGenerator.getGenerator(generatorType, trainingSamples, trainingLabels, batchSize, inputShape, dataAugmentation=False,  sequence=True)
validationGenerator = AVGenerator.getGenerator(generatorType, validationSamples, validationLabels, batchSize, inputShape,  sequence=True)

fullValidationGenerator = AVGenerator.getGenerator(generatorType, fullValidationSamples, fullValidationLabels, batchSize, inputShape,  sequence=True)


"""
Create Model
"""

model = AVClassifier.getModel(inputShape, modelType)

# model = AVClassifier.loadModel("/home/pablo/Documents/Datasets/FaceChannel_Outputs/AffWild2/Experiments/AffWIld2_Final/Models/AV_Exp/50k_BestAcc_2020-10-03 20:45:28.043769/Model")

"""
Train Model
"""
model = AVClassifier.train(model,experimentFolder,trainGenerator, validationGenerator, batchSize, epoches)

"""
Evaluate Model
"""

AVClassifier.evaluate(model, fullValidationGenerator, batchSize)