import os, sys
from glob import glob
import cv2 as cv
import numpy as np
import platform
import random

ims = {}
saveDir = "../preppedData/"
testSize = 0.2

for inputFile in glob("../originalData/BBBC010_v2_images/*"):
    if "w2" in inputFile:
        fname = inputFile[inputFile.find("w2")-4:inputFile.find("w2")-1]
        im = cv.imread(inputFile)

        if fname not in ims.keys():
            ims[fname] = [None, None]
        ims[fname][0] = im

        zerosArray = np.zeros_like(im)
        ims[fname][1] = zerosArray

wormLabels = {}

f = open("../trueFile.txt", "r")
for line in f:
    line = line.strip()
    fileName = line[:-2]
    label = int(line[-1])
    wormLabels[fileName] = label



for wormName in glob("../originalData/BBBC010_v1_foreground_eachworm/*"):
    wormim = cv.imread(wormName)
    wormName = os.path.basename(wormName)
    wormMask = wormim!=0
    fname = wormName[:3]
    currentim = ims[fname][1]
    currentim[wormMask] = wormLabels[wormName]

    #view = currentim.flatten()
    #counts = np.bincount(view)
    #print(counts[:4])
    ims[fname][1] = currentim

numOfIms = len(ims)

mixed_indices = [i for i in range(numOfIms)]
random.shuffle(mixed_indices)


i = 0
for key in ims.keys():
    if mixed_indices.index(i) >= numOfIms*testSize:
        inputFileName = saveDir + "input_train/" + key + ".png"
        outputFileName = saveDir + "output_train/" + key + ".png"
    else:
        inputFileName = saveDir + "input_test/" + key + ".png"
        outputFileName = saveDir + "output_test/" + key + ".png"
    inimresized = cv.resize(ims[key][0], (704, 512), interpolation = cv.INTER_AREA)
    outimresized = cv.resize(ims[key][1], (704, 512), interpolation = cv.INTER_AREA)

    cv.imwrite(inputFileName, inimresized)
    cv.imwrite(outputFileName, outimresized)

    i += 1
