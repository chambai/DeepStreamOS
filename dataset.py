# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 17:26:27 2019

@author: id127392
"""
from tensorflow.keras.datasets import cifar10, fashion_mnist
import util
import numpy as np
import cv2

isDataLoaded = False
x_train = None
y_train = None
y_train_umapped = None
x_test = None
y_test = None
y_test_unmapped = None

#--------------------------------------------------------------------------
def getFilteredData(isMap=True):  # if isMap is False, then mapped data will not be mapped and the original sub-classes will be returned
    x_train, y_train, x_test, y_test = getAllData()
    print(x_train.shape)
    print(x_test.shape)

    classes = util.getParameter('DataClasses')
    util.thisLogger.logInfo('Data classes to be used: %s'%(classes))
    x_train, y_train, x_test, y_test = filterData(x_train, y_train, x_test, y_test, classes)
    print(x_train.shape)
    print(x_test.shape)

    if isMap:
        # map data to different labels
        x_train, y_train, y_train_unmapped = mapClasses(x_train, y_train)
        x_test, y_test, y_test_unmapped = mapClasses(x_test, y_test)

    y_train = np.asarray([x[0] for x in y_train])
    y_test = np.asarray([x[0] for x in y_test])
    return x_train, y_train, x_test, y_test

#--------------------------------------------------------------------------
def filterData(x_train, y_train, x_test, y_test, classes):
    x_train, y_train = util.filterDataByClass(x_train, y_train, classes)
    x_test, y_test = util.filterDataByClass(x_test, y_test, classes)
        
    x_train = resize(x_train)
    x_test = resize(x_test)
    return x_train, y_train, x_test, y_test

#--------------------------------------------------------------------------
def getOutOfFilterData(isMap=False):
    x_train, y_train, x_test, y_test = getAllData()
    classes = util.getParameter('DataDiscrepancyClass')
    x_train, y_train, x_test, y_test = filterData(x_train, y_train, x_test, y_test, classes)

    if isMap:
        # map data to different labels
        x_train, y_train, y_train_unmapped = mapClasses(x_train, y_train)
        x_test, y_test, y_test_unmapped = mapClasses(x_test, y_test)

    y_train = np.asarray([x[0] for x in y_train])
    y_test = np.asarray([x[0] for x in y_test])
    return x_train, y_train, x_test, y_test

#--------------------------------------------------------------------------
def resetData():
    global isDataLoaded
    global x_train
    global y_train
    global y_train_unmapped
    global x_test
    global y_test
    global y_test_unmapped   
    
    isDataLoaded = False
    x_train = None
    y_train = None
    y_train_umapped = None
    x_test = None
    y_test = None
    y_test_unmapped = None
    isMapped = False
    
    
#--------------------------------------------------------------------------
def getAllData():
    global isDataLoaded
    global x_train
    global y_train
    global y_train_unmapped
    global x_test
    global y_test
    global y_test_unmapped   
    
    
    if isDataLoaded == False:
        datasetName = util.getParameter('DatasetName')
        if datasetName == 'cifar10':
            # get the cifar 10 dataset from Keras
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        elif datasetName == 'mnistfashion':
            # get the MNIST-Fashion dataset from Keras
            (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
            y_train = np.reshape(y_train, (y_train.shape[0],1))
            y_test = np.reshape(y_test, (y_test.shape[0],1))
        else:
            raise ValueError("Unhandled dataset name of %s"%(datasetName))
        
        # Normalise
        x_test = x_test.astype('float32')/255
        x_train = x_train.astype('float32')/255
        isDataLoaded = True
    
    return x_train, y_train, x_test, y_test

#-------------------------------------------------------------------------
def resize(x):
    out = []
    isReshaped, shape = getDataShape()
    
    if isReshaped:
        
        x_reshaped = []
        for i in range(len(x)):
            image = x[i]
            img = cv2.resize(image, (32,32), interpolation = cv2.INTER_NEAREST)
            x_reshaped.append(np.stack((img,)*3, axis=-1))
        x = np.asarray(x_reshaped)
        out = x

    else:
        out = x
    
    return np.asarray(out)

#-------------------------------------------------------------------------
def getDataShape():
    datasetName = util.getParameter('DatasetName')
    isReshaped = False
    if datasetName == 'cifar10':
        out =  (32,32,3)
    elif datasetName == 'mnistfashion':
        out =  (32,32,3)
        isReshaped = True
    else:
        out = (32,32,3)
    return isReshaped, out

#-------------------------------------------------------------------------
def mapClasses(x, y):
    global isMapped
    y_unmapped = y
    
    # maps data to higher level classes
    mapOriginalYValues = util.getParameter('MapOriginalYValues')
    
    # map data if mapOriginalYValues contains data
    if len(mapOriginalYValues) != 0:
        util.thisLogger.logInfo('Mapping classes: length of x data: %s. Length of y data: %s. Y data values: %s'%(len(x),len(y),np.unique(y)))
        mapNewYValues = np.asarray(util.getParameter('MapNewYValues'))
        mapNewNames = util.getParameter('MapNewNames')
    
        # check mapOriginalYValues and mapNewYValues are the same size
        if len(mapOriginalYValues) != len(mapNewYValues):
             raise ValueError("MapOriginalYValues array size (%s) does not match MapNewYValues array size (%s)"%(len(mapOriginalYValues), len(mapNewYValues)))

        # check distinct values of mapNewYValues match number of elements in mapNewNames
        distinctMapNewYValues = np.unique(mapNewYValues[mapNewYValues >= 0])
        if len(distinctMapNewYValues) != len(mapNewNames):
             raise ValueError("Distinct values of MapNewYValues (%s) does not match the number of elements in MapNewNames (%s)"%(len(distinctMapNewYValues), len(mapNewNames)))

        # if there's any -1 values in mapNewYValues, remove X and Y values for the corresponding class in mapOriginalYValues
        if -1 in mapNewYValues:
            # find out what elements in mapOriginalYValues the -1 corresponds to
            minusOneIndexes = np.where(mapNewYValues == -1)
            yValuesToRemove = mapOriginalYValues[minusOneIndexes]
            dataIndexesToRemove = np.in1d(y, yValuesToRemove).nonzero()[0]
            y = np.delete(y, dataIndexesToRemove, axis=0)
            y_unmapped = y
            x = np.delete(x, dataIndexesToRemove, axis=0)

        # change the Y values to the new higher level values
        for orig, new in zip(mapOriginalYValues, mapNewYValues):
            y = np.where(y==orig, new, y)
            
        isMapped = True
        
        util.thisLogger.logInfo('Mapped classes: length of x data: %s. Length of y data: %s. Y data values: %s'%(len(x),len(y),np.unique(y)))
    return x, y, y_unmapped

#-------------------------------------------------------------------------
def getDataMapAsString():
    # returns the new set of y values
    mapNewYValues = np.asarray(util.getParameter('MapNewYValues'))
    distinctMapNewYValues = np.unique(mapNewYValues[mapNewYValues >= 0])
    mapAsString = ''.join(map(str,distinctMapNewYValues))
    return mapAsString
    
#-------------------------------------------------------------------------
def getDataMap(unmappedClasses):
    # returns the new set of y values
    mapNewYValues = np.asarray(util.getParameter('MapNewYValues'))
    distinctMapNewYValues = np.unique(mapNewYValues[mapNewYValues >= 0])
    mapAsString = ''.join(map(str,distinctMapNewYValues))
    return mapAsString
    
    



    