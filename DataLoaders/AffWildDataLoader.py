import os
from tqdm import tqdm
import numpy
import random
import matplotlib.pyplot as plt
import tensorflow as tf

DATATYPE = {'Arousal_Frame': 'Arousal_Frame',
            'Valence_Frame': 'Valence_Frame',
            'Expression_Frame': 'Expression_Frame',

            "AV_Exp_Frame":"AV_Exp_Frame",
            "AV_Exp_Sequence": "AV_Exp_Sequence",

            'Arousal_Sequence': 'Arousal_Sequence',
            'Expression_Sequence': 'Expression_Sequence',


            "Expression_Test":  "Expression_Test",
            "Expression_Test_Sequence": "Expression_Test_Sequence"
            }

"""
Factory to get the right dataLoader
"""
def getData(videoDirectory, labelDirectory, type, shuffle=True, maxSamples=-1, splitName="", loadingBins=-1, histogramDirectory = "",
            sequenceSize = 10):

    classes = True

    if type == DATATYPE["Arousal_Frame"]:
        samples,labels = getArousalData_Frame(videoDirectory, labelDirectory, shuffle)

        if loadingBins > 0:
            samples,labels = balanceData(samples, labels, loadingBins)


    elif type == DATATYPE["Valence_Frame"]:

        samples, labels = getValenceData_Frame(videoDirectory, labelDirectory, shuffle)

        if loadingBins > 0:
            samples, labels = balanceData(samples, labels, loadingBins)

    elif type == DATATYPE["Expression_Frame"]:

        classes = True
        samples, labels = getExpressionData_Frame(videoDirectory, labelDirectory, shuffle)

        if loadingBins > 0:
            samples, labels = balanceExpressions(samples, labels, loadingBins)

        loadingBins = 7

    elif type == DATATYPE["AV_Exp_Frame"]:
        samples, labels = getAVExpData_Frame(videoDirectory, labelDirectory, shuffle)

        # classes = True
        # if loadingBins > 0:
        #     samples, labels = balanceAVExpressions(samples, labels, loadingBins)
        #
        # loadingBins = 7
        #
        histogramDirectory = ""

    elif type == DATATYPE["AV_Exp_Sequence"]:
        samples, labels = getAVExpData_Sequence(videoDirectory, labelDirectory, shuffle, sequenceSize)
        histogramDirectory = ""

    elif type == DATATYPE["Arousal_Sequence"]:
        samples, labels = getArousalData_Sequence(videoDirectory, labelDirectory, shuffle, sequenceSize)

    elif type == DATATYPE["Expression_Sequence"]:
        classes = True
        samples, labels = getExpressionData_Sequence(videoDirectory, labelDirectory, shuffle, sequenceSize)

        loadingBins = 7

    elif type == DATATYPE["Expression_Test"]:
        samples, labels = getTestSetVideos(videoDirectory, labelDirectory)

    elif type == DATATYPE["Expression_Test_Sequence"]:
        samples, labels = getTestSetVideosSequence(videoDirectory, labelDirectory, sequenceSize)

    if maxSamples > 0:
        samples = samples[0:maxSamples]
        labels = labels[0:maxSamples]


    print ("----------------------")
    print (splitName+" Samples: " + str(samples.shape))
    print(splitName+" Labels: " + str(labels.shape))
    print ("----------------------")

    if not histogramDirectory == "":
        if loadingBins < 0:
            loadingBins = 21
        createHistogram(labels, [], histogramDirectory, loadingBins, splitName, classes=classes)


    return samples,labels

"""
Utils
"""
def shuffleData(samples, labels):

    idx = numpy.random.choice(samples.shape[0], samples.shape[0], replace=False)
    x = samples[idx, ...]
    y = labels[idx, ...]

    return x, y



def balanceAVExpressions(samples, labels, totalBins):

    newSamples = []
    newLabels = []
    classLabels = labels[:, 2]
    # print ("SHape labels:" + str(classLabels.shape))
    # input("here")
    # classLabels = []
    # for a in labels[:, 2]:
    #     classLabels.append(numpy.argmax(a))

    bins = numpy.linspace(0, totalBins-1, totalBins)
    digitized = numpy.digitize(classLabels, bins)
    unique, counts = numpy.unique(digitized, return_counts=True)

    # print("Bins:" + str(bins))
    # print ("Counts:" + str(counts))
    # print ("Unique:" + str(unique))
    # input("here")
    maxBagValue = numpy.max(counts)

    bags = []
    for a in range(len(unique)):
        bags.append([])

    for a in range(len(samples)):
        # print ("digitized:" + str(digitized[a]-1))
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



def balanceExpressions(samples, labels, totalBins):

    newSamples = []
    newLabels = []

    classLabels = []
    for a in labels:
        classLabels.append(numpy.argmax(a))

    bins = numpy.linspace(0, totalBins-1, totalBins)
    digitized = numpy.digitize(classLabels, bins)
    unique, counts = numpy.unique(digitized, return_counts=True)

    # print("Bins:" + str(bins))
    # print ("Counts:" + str(counts))
    # print ("Unique:" + str(unique))
    # input("here")
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

def createHistogram(arousal, valence, directory,loadingBins, name, classes=False):



    if classes:
        classLabels = []
        for a in arousal:
            classLabels.append(numpy.argmax(a))
        arousal = classLabels

    createFolders(directory)
    directory = directory + "/DataVis/"
    createFolders(directory)

    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

    # We can set the number of bins with the `bins` kwarg
    if  not len(arousal) == 0:
        axs[0].hist(arousal, bins=loadingBins)
    if not len(valence) == 0:
        axs[1].hist(valence, bins=loadingBins)

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


def getExpressionData_Frame(videoDirectory, labelDirectory, shuffle):

    samples = []
    labels = []

    for fileName in tqdm(os.listdir(labelDirectory)):
        labelFile = open(labelDirectory + "/" + fileName)
        rowNumber = 0
        for line in labelFile:
            if rowNumber > 0:
                expression = line.split("\n")[0]
                # print ("Expression:" + str(expression))
                # input("here")
                if int(expression) >= 0:
                    fileNumber = str(rowNumber)

                    while not len(str(fileNumber)) == 5:
                        fileNumber = "0" + fileNumber

                    newFileName = videoDirectory + "/" + fileName.split(".")[0] + "/" + str(fileNumber) + ".jpg"
                    if os.path.exists(newFileName):
                        samples.append(newFileName)

                        # newLabel = numpy.zeros(7)
                        # newLabel[int(expression)] = 1
                        labels.append(int(expression))

            rowNumber +=1
        labelFile.close()

    labels = tf.keras.utils.to_categorical(labels, 7, dtype="float32")
    samples, labels = numpy.array(samples), numpy.array(labels)

    if shuffle:
        samples, labels = shuffleData(samples,labels)

    return samples, labels

def getAVExpData_Frame(videoDirectory, labelDirectories, shuffle):

    samples = []
    labels = []

    avDirectory, expDirectory = labelDirectories

    for fileName in tqdm(os.listdir(avDirectory)):


        if os.path.exists(expDirectory + "/" + fileName):
            avFile = open(avDirectory + "/" + fileName)
            expressionFile = open(expDirectory + "/" + fileName)
            expressionLines = expressionFile.readlines()
            rowNumber = 0
            for line in avFile:
                if rowNumber > 0:
                    valence, arousal = line.split(",")
                    expression = expressionLines[rowNumber]
                    expression = int(expression)

                    if float(valence) >= -1 and float(valence) <= 1 and expression > -1:
                        fileNumber = str(rowNumber)

                        while not len(str(fileNumber)) == 5:
                            fileNumber = "0" + fileNumber

                        newFileName = videoDirectory + "/" + fileName.split(".")[0] + "/" + str(fileNumber) + ".jpg"
                        if os.path.exists(newFileName):
                            samples.append(newFileName)

                            # print ("Exp:" + str(exp[0]))
                            # input("here")
                            # exp = numpy.zeros(7)
                            # exp[expression] = 1
                            label = numpy.asarray(arousal).astype(numpy.float32)
                            labels.append([float(arousal), float(valence), expression])

                rowNumber +=1
            avFile.close()
            expressionFile.close()

    samples, labels = numpy.array(samples), numpy.array(labels)

    if shuffle:
        samples, labels = shuffleData(samples,labels)

    return samples, labels



def getAVExpData_Sequence(videoDirectory, labelDirectories, shuffle, sequenceSize):

    samples = []
    labels = []

    avDirectory, expDirectory = labelDirectories

    for fileName in tqdm(os.listdir(avDirectory)):


        if os.path.exists(expDirectory + "/" + fileName):
            avFile = open(avDirectory + "/" + fileName)
            expressionFile = open(expDirectory + "/" + fileName)
            expressionLines = expressionFile.readlines()
            rowNumber = 0

            sequence = []
            allExpressions = []
            avgArousal = []
            avgValence = []


            for line in avFile:
                if rowNumber > 0:
                    valence, arousal = line.split(",")
                    expression = expressionLines[rowNumber]
                    expression = int(expression)

                    if float(valence) >= -1 and float(valence) <= 1 and expression > -1:
                        fileNumber = str(rowNumber)

                        while not len(str(fileNumber)) == 5:
                            fileNumber = "0" + fileNumber

                        newFileName = videoDirectory + "/" + fileName.split(".")[0] + "/" + str(fileNumber) + ".jpg"
                        if os.path.exists(newFileName):
                            if len(sequence) < sequenceSize:
                                sequence.append(newFileName)
                                allExpressions.append(int(expression))
                                avgArousal.append(float(arousal))
                                avgValence.append(float(valence))
                            else:
                                samples.append(sequence)
                                maxElement = max(allExpressions, key=allExpressions.count)

                                avgArousal = numpy.array(avgArousal).mean()
                                avgValence = numpy.array(avgValence).mean()

                                labels.append([float(avgArousal), float(avgValence), int(maxElement)])
                                # labels.append(int(maxElement))
                                sequence = []
                                sequence.append(newFileName)

                                allExpressions = []
                                avgArousal = []
                                avgValence = []

                                allExpressions.append(int(expression))
                                avgArousal.append(float(arousal))
                                avgValence.append(float(valence))

                            # samples.append(newFileName)

                            # print ("Exp:" + str(exp[0]))
                            # input("here")
                            # exp = numpy.zeros(7)
                            # exp[expression] = 1



                rowNumber +=1
            avFile.close()
            expressionFile.close()

    samples, labels = numpy.array(samples), numpy.array(labels)

    if shuffle:
        samples, labels = shuffleData(samples,labels)

    return samples, labels


def getExpressionData_Sequence(videoDirectory, labelDirectory, shuffle, sequenceSize):

    samples = []
    labels = []

    for fileName in tqdm(os.listdir(labelDirectory)):
        labelFile = open(labelDirectory + "/" + fileName)
        rowNumber = 0

        sequence = []
        allExpressions = []

        for line in labelFile:
            if rowNumber > 0:
                expression = line.split("\n")[0]

                fileNumber = str(rowNumber)

                while not len(str(fileNumber)) == 5:
                    fileNumber = "0" + fileNumber

                newFileName = videoDirectory + "/" + fileName.split(".")[0] + "/" + str(fileNumber) + ".jpg"
                if os.path.exists(newFileName):
                    if len(sequence) < sequenceSize:
                        sequence.append(newFileName)
                        allExpressions.append(int(expression))
                    else:
                        samples.append(sequence)
                        maxElement = max(allExpressions, key=allExpressions.count)
                        labels.append(int(maxElement))
                        sequence = []
                        sequence.append(newFileName)
                        allExpressions = []
                        allExpressions.append(int(expression))


            rowNumber +=1
        labelFile.close()

    labels = tf.keras.utils.to_categorical(labels, 7, dtype="float32")
    samples, labels = numpy.array(samples), numpy.array(labels)

    if shuffle:
        samples, labels = shuffleData(samples,labels)

    return samples, labels


def getTestSetVideos(videoDirectory, testSetDirectory):

    videoSamples = []
    labels = []

    testSetFile = open(testSetDirectory, "r")

    rowNumber = 0
    nonExistingFrames = 0
    for line in testSetFile:
        videoName = line.split("\n")[0]
        videoFolder = videoDirectory+"/"+videoName
        samplesThisVideo = []

        frames = os.listdir(videoFolder)
        # samplesThisVideo.extend(frames)
        # print("Video:" + str(videoFolder)+ " - Frames:" + str(len(frames)))
        # print ("Video directory:" + str(videoFolder))
        for f in frames:
            # print("Frame directory:" + str(videoFolder + "/" + f))
            if os.path.exists(videoFolder+"/"+f) and "jpg" in f:
                samplesThisVideo.append(videoFolder+"/"+f)
            else:
                nonExistingFrames += 1
        videoSamples.append(samplesThisVideo)
        labels.append(videoName)


        rowNumber +=1
    testSetFile.close()
    # print ("Non existing frames:" + str(nonExistingFrames))
    # input("here")
    return numpy.array(videoSamples), numpy.array(labels)


def getTestSetVideosSequence(videoDirectory, testSetDirectory, sequenceSize):

    videoSamples = []
    labels = []

    testSetFile = open(testSetDirectory, "r")

    rowNumber = 0
    nonExistingFrames = 0
    for line in testSetFile:
        videoName = line.split("\n")[0]
        videoFolder = videoDirectory+"/"+videoName
        samplesThisVideo = []

        frames = os.listdir(videoFolder)
        # print ("Frames:" + str(len(frames)))
        # samplesThisVideo.extend(frames)
        # print("Video:" + str(videoFolder)+ " - Frames:" + str(len(frames)))
        # print ("Video directory:" + str(videoFolder))

        for f in range(len(frames)):
            if f + 10 > len(frames):
                missingSamples = (f+10) - len(frames)
                sequence_f = frames[f:f+10]
                for s in range(missingSamples):
                    sequence_f.append(sequence_f[-1])
            else:
                sequence_f = frames[f:f+10]
            sequence = []
            for a in sequence_f:
                sequence.append(videoFolder+"/"+a)
            samplesThisVideo.append(sequence)


            # # print("Frame directory:" + str(videoFolder + "/" + f))
            # if os.path.exists(videoFolder+"/"+f) and "jpg" in f:
            #     samplesThisVideo.append(videoFolder+"/"+f)
            # else:
            #     nonExistingFrames += 1
        videoSamples.append(samplesThisVideo)
        labels.append(videoName)
        # print ("Inputs:" + str(len(samplesThisVideo)))
        #
        # input("here")
        rowNumber +=1
    testSetFile.close()
    # print ("Non existing frames:" + str(nonExistingFrames))
    # input("here")
    return numpy.array(videoSamples), numpy.array(labels)
