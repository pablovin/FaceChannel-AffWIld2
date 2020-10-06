import os
from DataLoaders import AffWildDataLoader
from Generators import AVGenerator
from Models import AVClassifier
from tqdm import tqdm
import numpy
import os

videosFolder = "/home/pablo/Documents/Datasets/affwild2/cropped_aligned"
testSetDirectory = "/home/pablo/Documents/Datasets/affwild2/Expr_Challenge_video_and_total_number_of_frames.txt"

saveDirectoryExp = "/home/pablo/Documents/Datasets/FaceChannel_Outputs/AffWild2/Experiments/AffWIld2_Final/ResultsCorrected/Expr Challenge-Track/FC-S"

model = "/home/pablo/Documents/Datasets/FaceChannel_Outputs/AffWild2/Experiments/AffWIld2_Final/Models/AV_Exp_Sequence/Sequence_Best_2020-10-03 22:03:17.618090/Model"

#Type of data, model and generator
dataType = AffWildDataLoader.DATATYPE["Expression_Test_Sequence"]
generatorType =AVGenerator.GENERATORTYPE["AVExp_FaceChannel"]

#Training Parameters
batchSize = 128
sequenceSize = 10

#Image parameters
inputShape = (sequenceSize, 112,112,3) # Image H, W, RGB


"""
Load Model
"""

model = AVClassifier.loadModel(model)

"""
Load data
"""
testSamples, testLabels = AffWildDataLoader.getData(videosFolder,testSetDirectory, dataType, sequenceSize)

for index, video in enumerate(testSamples):

    """Create a generator for this videos"""


    # input("here")
    videoGenerator = AVGenerator.getGenerator(generatorType, video, [numpy.zeros([len(video),1]),numpy.zeros([len(video),1]), numpy.zeros([len(video),7])],
                                                        batchSize, inputShape, sequence=True)

    predictions = AVClassifier.predict(model, videoGenerator)


    saveFileExp = open(saveDirectoryExp + "/" + testLabels[index] + ".txt", "a")
    saveFileExp.write("Neutral,Anger,Disgust,Fear,Happiness,Sadness,Surprise\n")

    # print ("SHape pred:" + str(predictions[1].shape))
    # print ("SHape pred:" + str(predictions[0].shape))
    # print ("SHape pred:" + str(predictions[2].shape))
    for a in range(len(predictions[0])):
        # print ("Shape:" + str(a[0].shape))
        # input("here")
        arousal,valence, categories = predictions[0][a][0], predictions[1][a][0], predictions[2][a]
        c = numpy.argmax(categories)
        # print ("prediction:" + str(c))
        # print ("Arousal:" + str(arousal))
        # print("Valence:" + str(valence))
        # print("Cat:" + str(c))
        # input("here")

        saveFileExp.write(str(c) + "\n")

    saveFileExp.close()

    print ("Video:" + str(index) + " - "+str(testLabels[index]) + " - Predictions:" + str(len(predictions[0])))

    # numpy.savetxt(saveFile+"/"+testLabels[index]+".txt", header="Neutral,Anger,Disgust,Fear,Happiness,Sadness,Surprise")
    # print ("Video: "+str(index)+" - Predictions shape:" + str(predictions.shape))
    # input("Here")
