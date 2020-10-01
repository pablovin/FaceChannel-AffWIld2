from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import hyperopt

import pickle

import hyperopt.plotting


from DataLoaders import AffWildDataLoader
from Models import AVClassifier
from Generators import AVGenerator
from datetime import datetime



import tensorflow as tf
from keras import backend as K

import numpy
import time

#Data location
#Local
# videoDirectory = "/home/pablo/Documents/Datasets/affwild2/cropped_aligned"
# trainingLabelDirectory = "/home/pablo/Documents/Datasets/affwild2/balancedAnnotationsTraining/"
# validationLabelDirectory = "/home/pablo/Documents/Datasets/affwild2/annotations-20200917T112933Z-001/annotations/VA_Set/Validation_Set"
# experimentFolder = "/home/pablo/Documents/Datasets/FaceChannel_Outputs/AffWild2/Optmization/"
#
# generalFolder = "/home/pablo/Documents/Datasets/FaceChannel_Outputs/AffWild2/Optmization/"

#Gcloud
videoDirectory = "/home/pablovin/dataset/affwild2/cropped_aligned"
trainingLabelDirectory = "/home/pablovin/dataset/affwild2/balancedAnnotationsTraining"
validationLabelDirectory = "/home/pablovin/dataset/affwild2/annotations/VA_Set/Validation_Set"
experimentFolder = "/home/pablovin/experiments/facechannel/optmization"

generalFolder = "/home/pablovin/experiments/facechannel/optmization"



timeNow = str(datetime.now())
experimentFolder = experimentFolder + "/" + timeNow

dataSetFolder = generalFolder + "/OptmizationDataset"

#Type of data, model and generator
dataType = AffWildDataLoader.DATATYPE["Arousal_Frame"]
modelType =AVClassifier.MODELTYPE["Arousal_Frame_FaceChannel_Optmizer"]
generatorType =AVGenerator.GENERATORTYPE["Arousal_FaceChannel"]

#Training Parameters
epoches = 10
batchSize = 64
maxSamplesTraining = 50000
maxSamplesValidation = 1000
trainingBins = 21

shuffle = True

#Image parameters
inputShape = (112,112,3) # Image H, W, RGB


"""
Load data
"""
trainingSamples, trainingLabels = AffWildDataLoader.getData(videoDirectory,trainingLabelDirectory, dataType, shuffle=shuffle, maxSamples=maxSamplesTraining, splitName="Training", loadingBins=trainingBins, histogramDirectory=experimentFolder)

validationSamples, validationLabels = AffWildDataLoader.getData(videoDirectory,validationLabelDirectory, dataType, shuffle=shuffle, maxSamples=maxSamplesValidation, splitName="Validation", histogramDirectory=experimentFolder)

# fullValidationSamples, fullValidationLabels = AffWildDataLoader.getData(videoDirectory,validationLabelDirectory, dataType, shuffle=shuffle, splitName="Full_Validation")

# input("here")
"""
Create Generator
"""

trainGenerator = AVGenerator.getGenerator(generatorType, trainingSamples, trainingLabels, batchSize, inputShape, dataAugmentation=True)
validationGenerator = AVGenerator.getGenerator(generatorType, validationSamples, validationLabels, batchSize, inputShape)



#Search space for the model

numMaxEvals = 40

space = hp.choice('a',
                  [
                      (hp.choice("denseLayer", [10, 100, 500, 1000]),
                       hp.uniform("initialLR", 0.0001, 0.5 ),
                       hp.choice("decay", [False, True]),
                       hp.uniform("momentum", 0.1, 0.9),
                       hp.choice("nesterov", [False, True]),
                       hp.choice("BatchSize", [16, 64, 128, 256, 512, 1024]),
                       hp.choice("SmallNetwork", [False,True]),
                       hp.choice("ShuntingInhibition", [False,True]),
                      )
                  ])

# space = hp.choice('a',
#                   [
#                       (
#                        hp.choice("SmallNetwork", [False,True]),
#                        hp.choice("ShuntingInhibition", [False,True]),
#                       )
#                   ])


"""
BEst:  (500, 'tanh', 0.043328431022224057, True, 0.788821718281539, False, 256)
Best:{'BatchSize': 4, 'a': 0, 'activationType': 1, 'decay': 0, 'denseLayer': 2, 'initialLR': 0.043328431022224057, 'momentum': 0.788821718281539, 'nesterov': 1}

BEst:  (500, 0.015476808651646383, True, 0.7408493385691893, True, 1024, False, False)
Best:{'BatchSize': 5, 'ShuntingInhibition': 0, 'SmallNetwork': 0, 'a': 0, 'decay': 1, 'denseLayer': 2, 'initialLR': 0.015476808651646383, 'momentum': 0.7408493385691893, 'nesterov': 1}

100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [7:19:47<00:00, 659.68s/trial, best loss: 0.6061173677444458]
"""
def objective(args):

    experimentFolder = "/home/pablo/Documents/Datasets/FaceChannel_Outputs/AffWild2/Optmization/"

    timeNow = str(datetime.now())
    experimentFolder = experimentFolder + "/" + timeNow


    """
    Create Model
    """
    model = AVClassifier.getModel(inputShape, modelType, args)

    """
    Train Model
    """
    print("Now training this: Args: " + str(args))
    model = AVClassifier.train(model, experimentFolder, trainGenerator, validationGenerator, batchSize, epoches, args, verbose=1)

    """
    Evaluate Model
    """
    scores = AVClassifier.evaluate(model, validationGenerator, batchSize)

    ccc = scores[1]

    if ccc < 0:
        ccc = 0

    inverseCCC = 1 - ccc

    saveFile = open(generalFolder + "/optimizationResume.txt", "a")
    saveLine = "Experiment: " + str(experimentFolder) + " , " + "Params: " + str(args) + "," + "CCC: " + str(ccc)+"\n"
    saveFile.write(saveLine)
    saveFile.close()
    # return wins
    print("Args: " + str(args) + "CCC:" + str(scores[1]))
    return {
        'loss': inverseCCC,
        'status': STATUS_OK,
        # -- store other results like this
        'eval_time': time.time(),
        'other_stuff': {'ccc': scores[1]},
    }


trials = Trials()
best = fmin(objective,
            space=space,
            algo=tpe.suggest,
            max_evals=numMaxEvals,
            trials=trials)



print("Saving the trials dataset:", experimentFolder)

pickle.dump(trials, open(dataSetFolder, "wb"))

print("Trials:", trials)
print("BEst: ", hyperopt.space_eval(space, best))
print("Best:" + str(best))

hyperopt.plotting.main_plot_history(trials, title="WinsHistory")

