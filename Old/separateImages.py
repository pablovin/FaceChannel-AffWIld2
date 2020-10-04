import os
import shutil

dataFrom = "/home/pablo/Documents/Datasets/FER/images/Training"
dataTo = "/home/pablo/Documents/Datasets/FER/FerPlusImages/All"


for folder in os.listdir(dataFrom):
    print ("Copying folder: " + str(folder))
    for file in os.listdir(dataFrom+"/"+folder):

        originalName = file.split(".")[0]

        newName = "fer"
        while not len(originalName) == 5:
            originalName = "0"+originalName
        newName = "fer00"+originalName+".png"

        shutil.copy(dataFrom+"/"+folder+"/"+file, dataTo+"/"+newName)
