import os
from DataLoaders import AffWildDataLoader
from Generators import EXPGenerator
from Models import EXPClassifier
from tqdm import tqdm
import numpy
import os

videosFolder = "/home/pablo/Documents/Datasets/affwild2/cropped_aligned"
testSetDirectory = "/home/pablo/Documents/Datasets/affwild2/expression_test_set.txt"

model = "/home/pablo/Documents/Datasets/affwild2/expression_test_set.txt"

saveFileDirectory = "/home/pablo/Documents/Datasets/FaceChannel_Outputs/AffWild2/Experiments/AffWIld2_Final/ResultFiles/Exp_Frame"

model = "/home/pablo/Documents/Datasets/FaceChannel_Outputs/AffWild2/Experiments/AffWIld2_Final/Models/Expression_Frame/Best/Model"

#Type of data, model and generator
dataType = AffWildDataLoader.DATATYPE["Expression_Test"]
generatorType =EXPGenerator.GENERATORTYPE["EXP_FaceChannel"]

#Training Parameters
batchSize = 1024

#Image parameters
inputShape = (112,112,3) # Image H, W, RGB


"""
Load Model
"""

model = EXPClassifier.loadModel(model)

"""
Load data
"""
testSamples, testLabels = AffWildDataLoader.getData(videosFolder,testSetDirectory, dataType)

for index, video in enumerate(testSamples):

    """Create a generator for this videos"""


    # input("here")
    videoGenerator = EXPGenerator.getGenerator(generatorType, video, numpy.zeros([len(video),1]),
                                                        batchSize, inputShape)

    predictions = EXPClassifier.predict(model, videoGenerator)

    saveFile = open(saveFileDirectory+"/"+testLabels[index]+".txt","a")
    saveFile.write("Neutral,Anger,Disgust,Fear,Happiness,Sadness,Surprise\n")
    for a in predictions:
        c = numpy.argmax(a)
        # print ("prediction:" + str(c))
        saveFile.write(str(c)+"\n")
    saveFile.close()

    print ("Video:" + str(index) + "- Predictions:" + str(len(predictions)))

    # numpy.savetxt(saveFile+"/"+testLabels[index]+".txt", header="Neutral,Anger,Disgust,Fear,Happiness,Sadness,Surprise")
    # print ("Video: "+str(index)+" - Predictions shape:" + str(predictions.shape))
    # input("Here")
