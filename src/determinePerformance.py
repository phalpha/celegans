import keras_segmentation
from keras_segmentation import predict
import time
import keras
from keras.models import *
from keras.layers import *
from matplotlib import pyplot as plt
from keras_segmentation.models.model_utils import get_segmentation_model
import cv2 as cv
from glob import glob
import numpy as np
from sys import platform
import os
import json
from keras_segmentation.train import find_latest_checkpoint
from keras_segmentation.models.resnet50 import get_resnet50_encoder

IMAGE_ORDERING = 'channels_last'
MERGE_AXIS = -1

def _unet( n_classes , encoder , l1_skip_conn=True,  input_height=416, input_width=608  ):

    img_input , levels = encoder( input_height=input_height ,  input_width=input_width )
    [f1 , f2 , f3 , f4 , f5 ] = levels

    o = f4

    o = ( ZeroPadding2D( (1,1) , data_format=IMAGE_ORDERING ))(o)
    o = ( Conv2D(512, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
    o = ( BatchNormalization())(o)

    o = ( UpSampling2D( (2,2), data_format=IMAGE_ORDERING))(o)
    o = ( concatenate([ o ,f3],axis=MERGE_AXIS )  )
    o = ( ZeroPadding2D( (1,1), data_format=IMAGE_ORDERING))(o)
    o = ( Conv2D( 256, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
    o = ( BatchNormalization())(o)

    o = ( UpSampling2D( (2,2), data_format=IMAGE_ORDERING))(o)
    o = ( concatenate([o,f2],axis=MERGE_AXIS ) )
    o = ( ZeroPadding2D((1,1) , data_format=IMAGE_ORDERING ))(o)
    o = ( Conv2D( 128 , (3, 3), padding='valid' , data_format=IMAGE_ORDERING ) )(o)
    o = ( BatchNormalization())(o)

    o = ( UpSampling2D( (2,2), data_format=IMAGE_ORDERING))(o)

    if l1_skip_conn:
        o = ( concatenate([o,f1],axis=MERGE_AXIS ) )

    o = ( ZeroPadding2D((1,1)  , data_format=IMAGE_ORDERING ))(o)
    o = ( Conv2D( 64 , (3, 3), padding='valid'  , data_format=IMAGE_ORDERING ))(o)
    o = ( BatchNormalization())(o)

    o = ( UpSampling2D( (2,2), data_format=IMAGE_ORDERING))(o)

    if l1_skip_conn:
        o = ( concatenate([ o ,img_input],axis=MERGE_AXIS )  )

    o = ( ZeroPadding2D( (1,1), data_format=IMAGE_ORDERING))(o)
    o = ( Conv2D( 64, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
    o = ( BatchNormalization())(o)

    o =  Conv2D( n_classes , (3, 3) , padding='same', data_format=IMAGE_ORDERING )( o )



    model = get_segmentation_model(img_input , o )


    return model


def resnet50_unet( n_classes ,  input_height=416, input_width=608 , encoder_level=3):

    model =  _unet( n_classes , get_resnet50_encoder ,  input_height=input_height, input_width=input_width  )
    model.model_name = "resnet50_unet"
    return model


def model_from_checkpoint_path(model, checkpoints_path ):

    assert ( os.path.isfile(checkpoints_path+"_config.json" ) ) , "Checkpoint not found."
    model_config = json.loads(open(  checkpoints_path+"_config.json" , "r" ).read())
    latest_weights = find_latest_checkpoint( checkpoints_path )
    assert ( not latest_weights is None ) , "Checkpoint not found."
    #print("loaded weights " , latest_weights )
    model.load_weights(latest_weights)
    return model


def predictImageType(image):
    image = cv.resize(image, (704, 512), interpolation = cv.INTER_AREA)
    model = resnet50_unet(3,  input_height=512, input_width=704  )


    checkpointsPath = "../checkpoints/resnet_unet_3"
    model_from_checkpoint_path(model, checkpointsPath)

    out = predict.predict(model = model, inp=image)

    out1d = out.flatten()
    counts = np.bincount(out1d)

    if counts[1] > counts[2]:
        print("alive")
        return "alive"
    else:
        print("dead")
        return "dead"

def wormLabelAccuracy(imageName, image, actualDirectory, wormDict):

    imageorigsize = image.shape
    image = cv.resize(image, (704, 512), interpolation = cv.INTER_AREA)
    model = resnet50_unet(3,  input_height=512, input_width=704  )


    checkpointsPath = "../checkpoints/resnet_unet_3"
    model_from_checkpoint_path(model, checkpointsPath)

    predictedim = predict.predict(model = model, inp=image)
    total = 0
    correct = 0
    cm = np.zeros((2,2))

    for key in wormDict.keys():
        #print(key)
        if key[:3] == imageName:

            wormim = cv.imread(actualDirectory + key)

            wormim = cv.cvtColor(wormim, cv.COLOR_BGR2GRAY)

            wormim = cv.resize(wormim, (imageorigsize[1], imageorigsize[0]), interpolation = cv.INTER_AREA)

            testim = np.zeros_like(predictedim)
            mask = wormim != 0
            testim[mask] = predictedim[mask]
            test1d = testim.flatten()
            counts = np.bincount(test1d)

            ifcorrect = False
            if len(counts == 1):
                counts = np.append(counts, 0)
            if len(counts == 2):
                counts = np.append(counts, 0)
            if counts[1] > counts[2] and wormDict[key] == 1:
                correct += 1
                cm[0][0]+=1
                ifcorrect = True
            elif counts[1] > counts[2] and wormDict[key] == 2:
                cm[0][1]+=1
            elif counts[1] < counts[2] and wormDict[key] == 1:
                cm[1][0]+=1
            elif counts[1] < counts[2] and wormDict[key] == 2:
                cm[1][1]+=1
                correct += 1
                ifcorrect = True
            total +=1
            #print(key, ifcorrect)
    print(correct, total)
    print(cm)
    return correct, total, cm

def visualizeImage(image):
    image[image==1] = 255
    image[image==2] = 127
    return image

def saveVisualizeImages(dir, savedir, predict):
    for file in glob(dir + "*"):
        image = cv.imread(file)
        if predict:
            checkpointsPath = "../checkpoints/resnet_unet_3"
            model = resnet50_unet(3,  input_height=512, input_width=704  )
            model_from_checkpoint_path(model, checkpointsPath)
            out = predict.predict(model = model, inp=image)
            im = visualizeImage(out)
            cv.imwrite("{save}/{name}".format(save = savedir, name = os.path.basename(file)), im)
        else:
            im = visualizeImage(image)
            cv.imwrite("{save}/{name}".format(save = savedir, name = os.path.basename(file)), im)


testdir = "../preppedData/input_test/"
testoutputdir = "../preppedData/output_test/"
actualdir = "../originalData/BBBC010_v1_foreground_eachworm/"

typeCorrect = 0
typeTotal = 0

#saveVisualizeImages(testoutputdir, "visualizeData/actual", False)
#saveVisualizeImages(testdir, "visualizeData/predicted", True)


for file in glob(testdir+"*"):
    image = cv.imread(file)
    file = os.path.basename(file)

    print(file)
    predictedType = predictImageType(image)
    actualType = "alive" if int(file[1:3]) <= 12 else "dead"
    if predictedType == actualType:
        typeCorrect += 1
    typeTotal += 1


print("Type Correct: ", typeCorrect)
print("Type Total: ", typeTotal)
print("Type Accuracy: ", 100*typeCorrect/typeTotal, "percent")

wormLabels = {}
f = open("../trueFile.txt", "r")
for line in f:
    line = line.strip()
    fileName = line[:-2]
    label = int(line[-1])
    wormLabels[fileName] = label

totaltotal = 0
totalcorrect = 0
confusion_matrix = np.zeros((2,2))

for file in glob(testdir+"*"):
    image = cv.imread(file)
    file = os.path.basename(file)
    currentcorrect, currenttotal, currentcm = wormLabelAccuracy(file[:3], image, actualdir, wormLabels)
    totalcorrect += currentcorrect
    totaltotal += currenttotal
    confusion_matrix += currentcm

print("Worm Correct: ", totalcorrect)
print("Worm Total: ", totaltotal)
print("Worm Accuracy: ", 100*totalcorrect/totaltotal, "percent")

print("Confusion Matrix: ", confusion_matrix)
