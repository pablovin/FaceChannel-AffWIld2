import os
from tqdm import tqdm
import numpy
import random
import matplotlib.pyplot as plt

DATATYPE = {'Arousal_Frame': 'Arousal_Frame',
            'Valence_Frame': 'Valence_Frame',
            'Arousal_Sequence': 'Arousal_Sequence'}

"""
Factory to get the right dataLoader
"""
def getData(videoDirectory, labelDirectory, type, shuffle=True, maxSamples=-1, splitName="", loadingBins=-1, histogramDirectory = "",
            sequenceSize = 10):

    if type == DATATYPE["Arousal_Frame"]:
        samples,labels = getArousalData_Frame(videoDirectory, labelDirectory, shuffle)

        if loadingBins > 0:
            samples,labels = balanceData(samples, labels, loadingBins)

    elif type == DATATYPE["Arousal_Sequence"]:
        samples, labels = getArousalData_Sequence(videoDirectory, labelDirectory, shuffle, sequenceSize)

    elif type == DATATYPE["Valence_Frame"]:

        samples, labels = getValenceData_Frame(videoDirectory, labelDirectory, shuffle)

        if loadingBins > 0:
            samples, labels = balanceData(samples, labels, loadingBins)

    if maxSamples > 0:
        samples = samples[0:maxSamples]
        labels = labels[0:maxSamples]


    print ("----------------------")
    print (splitName+" Samples: " + str(samples.shape))
    print(splitName+" Labels: " + str(labels.shape))
    print ("----------------------")

    if not histogramDirectory == "":
        createHistogram(labels, [], histogramDirectory, splitName)


    return samples,labels

"""
Utils
"""
def shuffleData(samples, labels):

    idx = numpy.random.choice(samples.shape[0], samples.shape[0], replace=False)
    x = samples[idx, ...]
    y = labels[idx, ...]

    return x, y

def balanceData(samples, labels, totalBins):

    newSamples = []
    newLabels = []

    bins = numpy.linspace(-1, 0.9, totalBins)
    digitized = numpy.digitize(labels, bins)
    unique, counts = numpy.unique(digitized, return_counts=True)

    maxBagValue = numpy.max(counts)

    bags = []
    for a in range(len(unique)):
        bags.append([])

    for a in range(len(samples)):
        bags[digitized[a]-1].append(a)

    for a in range(len(bags)):
        print ("Bag "  + str(a) + ":" + str(len(bags[a])))

    for a in range(len(bags)):
        if not len(bags[a]) == maxBagValue:
            while (len(bags[a])) < maxBagValue:
                difference = maxBagValue - len(bags[a])
                # print("Difference :" + str(difference))
                random.shuffle(bags[a])
                toBeAdded = bags[a][0:difference]
                random.shuffle(toBeAdded)
                # print("toBeAdded :" + str(len(toBeAdded)))
                bags[a].extend(toBeAdded)


    for a in range(len(bags)):
        print ("Bag "  + str(a) + ":" + str(len(bags[a])))
        newSamples.extend(samples[bags[a]])
        newLabels.extend(labels[bags[a]])

    # newSamples.extend(samples[bags[20]])
    # newLabels.extend(labels[bags[20]])

    # print("Example:" + str(labels[bags[0][0:10]]))
    # print ("Example:" + str(labels[bags[20][0:10]]))
    # print("Bag 20:" + str(len(bags[20])))
    # createHistogram(newLabels, [], "/home/pablo/Documents/Datasets/FaceChannel_Outputs/AffWild2/Experiments/", "TrainingTest")
    # input("here")
    newSamples, newLabels=     numpy.array(newSamples), numpy.array(newLabels)
    newSamples, newLabels = shuffleData(newSamples, newLabels)
    return  newSamples, newLabels

def createFolders(folder):

    if not os.path.exists(folder):
        os.mkdir(folder)

def createHistogram(arousal, valence, directory, name):

    createFolders(directory)
    directory = directory + "/DataVis/"
    createFolders(directory)

    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

    # We can set the number of bins with the `bins` kwarg
    if  not len(arousal) == 0:
        axs[0].hist(arousal, bins=22)
    if not len(valence) == 0:
        axs[1].hist(valence, bins=22)

    plt.savefig(directory + "/_"+str(name)+"_dataDistribution_.png")

    plt.clf()

"""
Different dataLoaders
"""

def getArousalData_Sequence(videoDirectory, labelDirectory, shuffle, sequenceSize):

    samples = []
    labels = []

    for fileName in tqdm(os.listdir(labelDirectory)):
        labelFile = open(labelDirectory + "/" + fileName)
        rowNumber = 0

        sequence = []
        averageArousal = []


        for line in labelFile:
            if rowNumber > 0:
                valence, arousal = line.split(",")
                if float(arousal) >= -1 and float(arousal) <= 1:
                    fileNumber = str(rowNumber)

                    while not len(str(fileNumber)) == 5:
                        fileNumber = "0" + fileNumber

                    newFileName = videoDirectory + "/" + fileName.split(".")[0] + "/" + str(fileNumber) + ".jpg"
                    if os.path.exists(newFileName):

                        if len(sequence) < sequenceSize:
                            sequence.append(newFileName)
                            averageArousal.append(float(arousal))

                        else:
                            samples.append(sequence)
                            avgArousal = numpy.array(averageArousal).mean()

                            labels.append(avgArousal)

                            sequence = []
                            sequence.append(newFileName)

                            averageArousal = []

                            averageArousal.append(float(arousal))


            rowNumber +=1
        labelFile.close()

    samples, labels = numpy.array(samples), numpy.array(labels)

    if shuffle:
        samples, labels = shuffleData(samples,labels)

    return samples, labels

def getArousalData_Frame(videoDirectory, labelDirectory, shuffle):

    samples = []
    labels = []

    for fileName in tqdm(os.listdir(labelDirectory)):
        labelFile = open(labelDirectory + "/" + fileName)
        rowNumber = 0
        for line in labelFile:
            if rowNumber > 0:
                valence, arousal = line.split(",")
                if float(arousal) >= -1 and float(arousal) <= 1:
                    fileNumber = str(rowNumber)

                    while not len(str(fileNumber)) == 5:
                        fileNumber = "0" + fileNumber

                    newFileName = videoDirectory + "/" + fileName.split(".")[0] + "/" + str(fileNumber) + ".jpg"
                    if os.path.exists(newFileName):
                        samples.append(newFileName)
                        labels.append(float(arousal))

            rowNumber +=1
        labelFile.close()

    samples, labels = numpy.array(samples), numpy.array(labels)

    if shuffle:
        samples, labels = shuffleData(samples,labels)

    return samples, labels

def getValenceData_Frame(videoDirectory, labelDirectory, shuffle):

    samples = []
    labels = []

    for fileName in tqdm(os.listdir(labelDirectory)):
        labelFile = open(labelDirectory + "/" + fileName)
        rowNumber = 0
        for line in labelFile:
            if rowNumber > 0:
                valence, arousal = line.split(",")
                if float(valence) >= -1 and float(valence) <= 1:
                    fileNumber = str(rowNumber)

                    while not len(str(fileNumber)) == 5:
                        fileNumber = "0" + fileNumber

                    newFileName = videoDirectory + "/" + fileName.split(".")[0] + "/" + str(fileNumber) + ".jpg"
                    if os.path.exists(newFileName):
                        samples.append(newFileName)
                        labels.append(float(valence))

            rowNumber +=1
        labelFile.close()

    samples, labels = numpy.array(samples), numpy.array(labels)

    if shuffle:
        samples, labels = shuffleData(samples,labels)

    return samples, labels