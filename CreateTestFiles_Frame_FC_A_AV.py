import os
from DataLoaders import AffWildDataLoader
from Generators import AVGenerator
from Models import AVClassifier
from tqdm import tqdm
import numpy
import os

videosFolder = "/home/pablo/Documents/Datasets/affwild2/cropped_aligned"
testSetDirectory = "/home/pablo/Documents/Datasets/affwild2/VA_Challenge_video_and_total_number_of_frames.txt"

saveFileDirectory = "/home/pablo/Documents/Datasets/FaceChannel_Outputs/AffWild2/Experiments/AffWIld2_Final/ResultsCorrected/VA Challenge-Track/FC_A"

model = "/home/pablo/Documents/Datasets/FaceChannel_Outputs/AffWild2/Experiments/AffWIld2_Final/Models/Arousal_Frame/2020-10-02 18:20:18.755732/Model"

#Type of data, model and generator
dataType = AffWildDataLoader.DATATYPE["Expression_Test"]
generatorType =AVGenerator.GENERATORTYPE["Arousal_FaceChannel"]

#Training Parameters
batchSize = 1024

#Image parameters
inputShape = (112,112,3) # Image H, W, RGB


"""
Load Model
"""

model = AVClassifier.loadModel(model)

"""
Load data
"""
testSamples, testLabels = AffWildDataLoader.getData(videosFolder,testSetDirectory, dataType)

for index, video in enumerate(testSamples):

    """Create a generator for this videos"""


    # input("here")
    videoGenerator = AVGenerator.getGenerator(generatorType, video, numpy.zeros([len(video),1]),
                                                        batchSize, inputShape)

    predictions = AVClassifier.predict(model, videoGenerator)

    saveFile = open(saveFileDirectory+"/"+testLabels[index]+".txt","a")
    saveFile.write("valence, arousal\n")
    for a in predictions:
        # print ("Shape:" + str(a.shape))
        arousal,valence = a[0], 0
        # print ("prediction:" + str(c))
        saveFile.write(str(valence)+","+str(arousal)+"\n")
    saveFile.close()

    print("Video:" + str(index) + " - " + str(testLabels[index]) + " - Predictions:" + str(len(predictions)))

    # numpy.savetxt(saveFile+"/"+testLabels[index]+".txt", header="Neutral,Anger,Disgust,Fear,Happiness,Sadness,Surprise")
    # print ("Video: "+str(index)+" - Predictions shape:" + str(predictions.shape))
    # input("Here")
