# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 12:05:39 2019

@author: id127392
"""
import util
from skmultiflow.data import DataStream
import extract
import numpy as np
import datetime

moduleName = None
module = None
results = None
instanceProcessingStartTime = None

#----------------------------------------------------------------------------
def loadModule(modName, dsData=None):
    global moduleName, module, results
    results = None
    moduleName = modName
    module = __import__(moduleName, fromlist=[''])
    if hasattr(module, 'setData'):
        module.setData(dsData)

#----------------------------------------------------------------------------
def setup(flatActivations, y_train):
    global module, moduleName
    util.thisLogger.logInfo("---------- start of %s setup for activaton analysis----------"%(moduleName))
    module.setupAnalysis(flatActivations, y_train)
    util.thisLogger.logInfo("----------- end of %s setup for activaton analysis----------\n"%(moduleName))

#----------------------------------------------------------------------------
def setupCompare(dnn, x_train, y_train):
    global module, moduleName
    util.thisLogger.logInfo("---------- start of %s setup for comparison----------"%(moduleName))
    module.setupAnalysis(dnn, x_train, y_train)
    util.thisLogger.logInfo("----------- end of %s setup for comparison----------\n"%(moduleName))

# ----------------------------------------------------------------------------
def startDataInputStream(dnnModel, simPrediction, maxClassValues1, maxClassValues2, unseenData):

    util.thisLogger.logInfo("\n---------- start of data input stream ----------")
    returnUnseenInstances = startDataInputStream_DataStream(dnnModel, simPrediction, maxClassValues1,maxClassValues2, unseenData)
    util.thisLogger.logInfo("----------- end of data input stream ----------\n")

    return returnUnseenInstances

# ----------------------------------------------------------------------------
def startDataInputStream_DataStream(dnnModel, simPrediction, maxClassValues1, maxClassValues2, unseenData):
    global module, moduleName, instanceProcessingStartTime
    batchLength = util.getParameter('StreamBatchLength')
    analysisName = util.getParameter('AnalysisModule')

    # split unseen instance object list into batches
    unseenDataBatches = [unseenData[i:i + batchLength] for i in range(0, len(unseenData), batchLength)]

    dataInstances = np.array([x.instance.flatten() for x in unseenData],dtype=float)

    if dnnModel == None:
        # We already have the dnn predicts from the activations file
        dnnPredicts = np.array([x.predictedResult for x in unseenData], dtype=float)
        stream = DataStream(dataInstances, dnnPredicts)
    else:
        # pass in zeros for Y data as the dnn predicts will be done later
        stream = DataStream(dataInstances, np.zeros(len(dataInstances)))

    instanceProcessingStartTime = datetime.datetime.now()
    returnUnseenInstances = []
    batchIndex = 1
    instIndex = 0
    while stream.has_more_samples():
        util.thisLogger.logInfo('batch %s' % (batchIndex))

        batchDataInstances, batchDnnPredicts = stream.next_sample(batchLength)  # get the next sample

        # get the original data shape and shape the batch data instances
        dataShape = unseenData[0].instance.shape
        batchDataInstances = np.reshape(batchDataInstances,
                                        (len(batchDataInstances), dataShape[1], dataShape[2], dataShape[3]))

        if unseenData[instIndex].reducedInstance.shape[0] == 0 or unseenData[instIndex].reducedInstance[0] == None:
            batchDnnPredicts = getPredictions(dnnModel, batchDataInstances)

            if 'compare' in analysisName:
                batchActivations = batchDataInstances
            else:
                batchActivations = online_processInstances(batchDataInstances, dnnModel, maxClassValues1)
        else:
            # data has already been reduced/normalized etc...
            batchActivations = np.array([u.reducedInstance for u in unseenDataBatches[batchIndex-1]],dtype=float)


        if 'dnn' in moduleName:
            batchDataInstances = [u.instance for u in unseenDataBatches[batchIndex-1]]
            dataShape = batchDataInstances[0].shape
            batchDataInstances = np.reshape(batchDataInstances,
                                            (len(batchDataInstances), dataShape[1], dataShape[2], dataShape[3]))

            batchDiscrepResult = module.processUnseenStreamBatch(batchDataInstances, batchActivations, batchDnnPredicts)
        else:
            batchDiscrepResult = module.processUnseenStreamBatch(batchActivations, batchDnnPredicts)

        for i, (act, dnnPredict, res) in enumerate(zip(batchActivations, batchDnnPredicts, batchDiscrepResult)):
            unseenData[instIndex].reducedInstance = np.array(act)
            unseenData[instIndex].predictedResult = dnnPredict
            unseenData[instIndex].discrepancyResult = res
            returnUnseenInstances.append(unseenData[instIndex])
            instIndex += 1

        batchIndex += 1

    if hasattr(module, 'endOfUnseenStream'):
        module.endOfUnseenStream()

    # # print out string of results for debug purposes
    # classStr = ''
    # discrepancyStr = ''
    # for inst in returnUnseenInstances:
    #     classStr += ''.join(str(inst.correctResult))
    #     discrepancyStr += ''.join(str(inst.discrepancyResult))

    return np.asarray(returnUnseenInstances)

# ----------------------------------------------------------------------------
def online_processInstances(instances, dnnModel, maxClassValues1):
    # processes the instances by getting the predictions and activations from the DNN and reducing them.
    global instanceProcessingStartTime
    instanceProcessingStartTime = datetime.datetime.now()
    util.thisLogger.logInfo('Start of instance processing, %s' % (len(instances)))

    # extract
    flatActivations, numLayers = extract.getActivations(dnnModel, [instances])
    del instances

    # Normalize
    flatActivations = flatActivations / maxClassValues1

    return flatActivations
#
# ----------------------------------------------------------------------------
def getPredictions(dnnModel, instances):
    predictions = np.argmax(dnnModel.predict(instances), axis=1)

    # The DNN is trained to output 0 or 1 only.
    mapNewYValues = util.getParameter('MapNewYValues')
    if len(mapNewYValues) == 0:
        # get the original classes it was trained on and transform the outputs
        classes = util.getParameter('DataClasses')
        util.thisLogger.logInfo('Data classes to be used: %s' % (classes))
        predictions = util.transformZeroIndexDataIntoClasses(predictions, classes)
    else:
        # get the mapped values the DNN was trained on
        mapNewYValues = np.unique(mapNewYValues)
        mapNewYValues = mapNewYValues[mapNewYValues >= 0]
        util.thisLogger.logInfo('Mapped class values to be used: %s' % (mapNewYValues))

    return predictions


