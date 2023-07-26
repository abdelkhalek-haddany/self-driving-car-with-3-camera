from sklearn.utils import shuffle, random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import matplotlib.image as mpimg
from imgaug import augmenters as iaa
import cv2
import random
from keras.models import Sequential
from keras.layers import Convolution2D,Flatten,Dense
from keras.optimizers import Adam

def getName(filePath):
    return filePath.split('\\')[-1]

def importDataInfo(path):
    columns = ['Center', 'Left', 'Right', 'Steering', 'Throttle', 'Brake', 'Speed']
    data = pd.read_csv(os.path.join(path, 'driving_log.csv'), names=columns)
    print(data.head())
    #print(data['Center'][0])
    #print(getName(data['Center'][0]))
    data['Center'] = data['Center'].apply(getName)
    #print(data.head())
    #print('Total Images Imported : ', data.shape[0]) #7893
    return data

def balanceData(data, display=True):
    nBin = 31
    samplesPerBin = 800
    hist, bins = np.histogram(data['Steering'], nBin)

    if display:
        print("Affichage de histogram ...")
        center = (bins[:-1] + bins[1:]) * 0.5
        plt.bar(center, hist, width=0.06)
        plt.plot((np.min(data['Steering']), np.max(data['Steering'])), (samplesPerBin, samplesPerBin))
        plt.show()
        print("Affichage est fermer")
    removeindexList = []
    for j in range(nBin):
        binDataList = []
        for i in range(len(data['Steering'])):
            if data['Steering'][i] >= bins[j] and data['Steering'][i] <= bins[j + 1]:
                binDataList.append(i)
        binDataList = shuffle(binDataList)
        binDataList = binDataList[samplesPerBin:]
        removeindexList.extend(binDataList)

    print('Removed Images : ', len(removeindexList))
    data.drop(data.index[removeindexList], inplace=True)
    print('Remaining Images : ', len(data))


    if display:
        print("Affiche de histogramme cleaning")
        hist, _ = np.histogram(data['Steering'], (nBin))
        plt.bar(center, hist, width=0.06)
        plt.plot((np.min(data['Steering']), np.max(data['Steering'])), (samplesPerBin, samplesPerBin))
        plt.show()
        print("Fermer d'affichage")
    return data

def loadData(path, data):
    print("Preaparing ...")
    print("Loading data ...")
    imagesPath = []
    steering = []
    for i in range(len(data)):
        indexed_data = data.iloc[i]
        #imagesPath.append(f'{path}\IMG\{indexed_data[0]}')
        imagesPath.append(os.path.join(path, 'IMG', indexed_data[0]))
        steering.append(float(indexed_data[3]))
    print("\timages Path ... Done")
    imagesPath = np.asarray(imagesPath)
    print("\tsteering ... Done")
    steering = np.asarray(steering)
    print("End Preparing .")
    return imagesPath, steering

def augmentImage(imgPath, steering):
    img = mpimg.imread(imgPath)

    ## PAN
    if np.random.rand() < 0.5:
        pan = iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})
        img = pan.augment_image(img)

    ## ZOOM
    if np.random.rand() < 0.5:
        zoom = iaa.Affine(scale=(1, 1.2))
        img = zoom.augment_image(img)

    ## BRITHTNESS
    if np.random.rand() < 0.5:
        brightness = iaa.Multiply((0.2, 1.2))
        img = brightness.augment_image(img)

    ## FLIP
    if np.random.rand() < 0.5:
        img = cv2.flip(img, 1)

    steering = -steering
    return img, steering

def preProcessing(img):
    #Cropping
    img = img[60:135,:,:]
    #YUV Colorspace
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    #Blur
    img = cv2.GaussianBlur(img, (3, 3), 0)
    #Resize
    img = cv2.resize(img, (200, 66))
    img = img / 255
    return img

def batchGen(imagesPath, steeringList, batchSize, trainFlag):
    while True:
        imgBatch = []
        steeringBatch = []

        for i in range(batchSize):
            index = random.randint(0, len(imagesPath) - 1)
            if trainFlag:
                imgPath = imagesPath[index]
                steering = steeringList[index]
                img, steering = augmentImage(imgPath, steering)
            else:
                img = mpimg.imread(imagesPath[index])
                steering = steeringList[index]
            img = preProcessing(img)
            imgBatch.append(img)
            steeringBatch.append(steering)
        yield np.asarray(imgBatch), np.asarray(steeringBatch)

def creatModel():
    model = Sequential()

    model.add(Convolution2D(24, (5,5), (2, 2), input_shape=(66, 200, 3), activation='elu'))
    model.add(Convolution2D(36, (5,5), (2, 2), activation='elu'))
    model.add(Convolution2D(48, (5,5), (2, 2), activation='elu'))
    model.add(Convolution2D(64, (3,3), activation='elu'))
    model.add(Convolution2D(64, (3,3), activation='elu'))

    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))

    model.compile(Adam(learning_rate=0.0001), loss='mse')
    return model
