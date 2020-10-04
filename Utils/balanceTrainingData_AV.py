import os
import math

videoDirectory = "/home/pablo/Documents/Datasets/affwild2/cropped_aligned"

trainingLabelDirectory = "/home/pablo/Documents/Datasets/affwild2/annotations-20200917T112933Z-001/annotations/VA_Set/Training_Set"
expressionLabels = "/home/pablo/Documents/Datasets/affwild2/annotations-20200917T112933Z-001/annotations/EXPR_Set/Training_Set"

balancedTraining = "/home/pablo/Documents/Datasets/affwild2/balancedAnnotationsTraining/"


framesOut = 0
for file in os.listdir(trainingLabelDirectory):
    print ("Reading file:" + str(file))

    if os.path.exists(expressionLabels+"/"+file):
        labelFile = open(trainingLabelDirectory+"/"+file)
        expressionFile = open(expressionLabels+"/"+file)
        expressionLines = expressionFile.readlines()

        rowNumber = 0

        newFile = open(balancedTraining+"/"+file, "a")
        newFile.write("valence,arousal\n")
        for line in labelFile:
            if rowNumber > 0:
                valence,arousal = line.split(",")
                valence = float(valence)
                arousal = float(arousal)
                expression = expressionLines[rowNumber]
                expression = float(expression)

                if float(valence) >= -1 and float(valence) <= 1:
                    fileNumber = str(rowNumber)
                    while not len(str(fileNumber)) == 5:
                        fileNumber = "0"+fileNumber

                    fileName = videoDirectory + "/" + file.split(".")[0] + "/" + str(fileNumber) + ".jpg"
                    if os.path.exists(fileName):
                        # Check for affective consistence
                        """
                        6000 frames labeled ”happy” but with negative valence. • 
                        13000 frames labeled ”sad” but
                        with positive valence.
                        • 121000
                        frames
                        labeled ”neutral” but
                        with high valence
                        arousal
                        norm
                       """
                        affConsistence = True
                        if expression == 4 and valence < 0:
                            affConsistence = False
                        #
                        if expression == 5 and valence > 0:
                            affConsistence = False
                        #
                        score = math.sqrt(valence*valence + arousal*arousal)
                        if expression == 0 and not score > 0.5:
                            affConsistence = False

                        if affConsistence:
                            newFile.write(str(valence)+","+str(arousal)+"\n")
                        else:
                            framesOut +=1
                            newFile.write(str(-5) + "," + str(-5)+"\n")
                    else:
                        framesOut +=1
                        newFile.write(str(-5) + "," + str(-5)+"\n")
                else:
                    framesOut += 1
                    newFile.write(str(-5) + "," + str(-5)+"\n")



            rowNumber +=1

        labelFile.close()
        expressionFile.close()
        newFile.close()

print ("Frames out: " + str(framesOut))