from py4j.java_gateway import JavaGateway
from py4j.java_collections import ListConverter
import sys
import threading
import util
import uuid
import multiprocessing
import numpy as np
import time
import os

dsData = {}

numCpus = multiprocessing.cpu_count()
moaGatewayStarted = False
gateway = None
process = False
results = None
gatewayDict = {}
streamList = {}
clustererList = {}
batchSize = 0
numProcessedInst = 0

def setData(inDsData):
    # optional - sets peripheral data
    global dsData
    dsData = inDsData

def setupAnalysis(act_train_batch, y_train_batch):
    # batch from the stream is provided to this function
    # create an MCOD clusterer for each class
    # add the training data to the MCOD clusterers with the activation data
    global dsData, results, gatewayDict
    gatewayDict = {}
    init()

    util.killMoaGateway() # kill any instances of moagateway

    path = os.path.join(*util.getFolderList(__file__)[:-1])
    moagatewaybatfilename = os.path.join(path,'moagatewaycmd.bat')
    util.startMoaGateway(moagatewaybatfilename)
    time.sleep(1) # wait otherwise sometimes the gateway is not ready when it is time to use it.

    dataSourceName = 'add'
    clustererName = 'mcod'
    results = None
    gatewayDict = JavaGateway()
    setupAnalysis_MoaFile(act_train_batch, y_train_batch, dataSourceName, clustererName)

def processUnseenStreamBatch(act_unseen_batch, dnnPredict_batch):
    global dsData, results, batchSize, process, numProcessedInst
    result = []
    numProcessedInst = 0
    batchSize = len(act_unseen_batch)
    results = []

    # start the thread to process the streams so that new instances get clustered
    process = True
    thread1 = threading.Thread(target=processStreamInstances, args=(act_unseen_batch, False, True, True), daemon=True)
    thread1.start()
    batchProcessObjList(act_unseen_batch, dnnPredict_batch)
    thread1.join()

    # get result
    for r in results:
        if r[2] == 'OUTLIER':
            result.append('D')
        elif r[2] == 'NOT_OUTLIER':
            result.append('N')
        elif r[2] == 'NO_OUTLIERS_REPORTED':
            result.append('U')

    return result

def endOfUnseenStream():
    global streamList
    util.killMoaGateway()
    for streamName in streamList.keys():
        tFilename = 'output/trainingactivations_%s.csv' % (streamName)
        uFilename = 'output/unseenactivations_%s.csv' % (streamName)
        if os.path.exists(tFilename):
            os.remove(tFilename)
        if os.path.exists(uFilename):
            os.remove(uFilename)

def init():
    global dsData, moaGatewayStarted, gateway, process, results, gatewayDict, streamList, clustererList, batchSize, numProcessedInst

    moaGatewayStarted = False
    gateway = None
    process = False
    results = None
    gatewayDict = {}
    streamList = {}
    clustererList = {}
    batchSize = 0
    numProcessedInst = 0


def processStreamInstances(act_unseen_batch, processOnce, batchProcessUnseenInstances, saveOutlierResults):
    global dsData, moaGatewayStarted, gateway, process, results, gatewayDict, streamList, clustererList, batchSize, numProcessedInst

    numInstances = len(act_unseen_batch)
    clustererName = 'mcod'

    if processOnce:
        for i in streamList.keys():
            processStreamInstance(i, clustererName, numInstances, batchProcessUnseenInstances, saveOutlierResults)
    else:
        process = True
        while process:
            for i in streamList.keys():
                processStreamInstance(i, clustererName, numInstances, True,
                                      saveOutlierResults)

    util.thisLogger.logInfo('processing stopped')

# ----------------------------------------------------------------------------
def checkConnection():
    global gatewayDict
    nTrys = 10
    for i in range(nTrys):
        try:
            # create the attribute names for the flat data
            attributeNames = [str(i) for i in range(10)]
            # this is the first time we attemp to connect to moa gateway
            jAttributeNames = ListConverter().convert(attributeNames, gatewayDict._gateway_client)
            break
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            util.thisLogger.logInfo(
                'problem connecting to gateway: %s, %s, %s, %s, %s' % (e, exc_type, exc_obj, exc_tb, exc_tb.tb_lineno))
            util.thisLogger.logInfo('trying again (try %s of %s)...' % (i, nTrys))
            if nTrys - 1 == i:
                raise Exception("problem connecting to gateway: %s: %s" % (exc_tb.tb_lineno, e))


# ----------------------------------------------------------------------------
def getAttributeNames(flatActivations):
    global gatewayDict
    # create the attribute names for the flat data
    attributeNames = [str(i) for i in range(0, len(flatActivations[0]))]
    jAttributeNames = ListConverter().convert(attributeNames, gatewayDict._gateway_client)
    return jAttributeNames


# ----------------------------------------------------------------------------
def setupAnalysis_MoaFile(flatActivations, y_train, dataSourceName, clustererName):
    global gatewayDict, streamList, clustererList
    mcod_clusterCreation = 'newjar'  # make parameter # thread or newjar

    classes = np.unique(y_train).astype(int)
    util.thisLogger.logInfo("\n%s classes found in entire training data" % (classes))

    # setup MOA java streams and clusters
    jAttributeNames = getAttributeNames(flatActivations)

    # get data into java
    for dataClass in classes:
        # create stream
        stream = createStream(jAttributeNames, dataSourceName)
        streamList[str(dataClass)] = stream
        # create clusterer
        clusterer = createClusterer(stream, dataClass)
        clustererList[str(dataClass)] = clusterer
        util.thisLogger.logInfo("Created stream and cluster " + str(dataClass))

        # add training instances to class stream from csv file
        javaFile = os.path.join(os.path.join(*util.getFolderList(__file__)[:-1]), "output/trainingactivations_%s.csv" % (dataClass))

        strDataClass = str(dataClass)

        if mcod_clusterCreation != 'thread':
            stream = streamList[strDataClass]
            gatewayDict.entry_point.Moa_Clusterers_Outliers_Mcod_AddCsvDataToStream(stream, javaFile)
            util.thisLogger.logInfo(
                "Training instance " + str(dataClass) + ": class " + str(dataClass) + ", added to stream " + str(
                    dataClass))

    # process the instances on each stream so they are clustered
    for i in streamList.keys():
        util.thisLogger.logInfo("processing stream instance " + str(i))
        processStreamInstance(i, clustererName, len(flatActivations))


# ----------------------------------------------------------------------------
def addUnseenDataToStreams_MoaFile(stream, javaFile, ids, streamName):
    global gatewayDict
    # get data into java
    jIds = ListConverter().convert(ids, gatewayDict._gateway_client)

    gatewayDict.entry_point.Moa_Clusterers_Outliers_Mcod_AddCsvDataToStream(stream, javaFile, jIds)


# ----------------------------------------------------------------------------
def setAnalysisParameters_Mcod(clusterer, k, radius, windowSize, dataClass):
    global gatewayDict
    params = []
    params = gatewayDict.entry_point.Moa_Clusterers_Outliers_MCOD_SetParameters(clusterer, k, radius,
                                                                                                windowSize)
    util.thisLogger.logInfo('MCOD Parameters: %s' % (params))
    return params.split(',')

# ----------------------------------------------------------------------------
def saveToArffFormat(csvFileNames, arffFileNames, classColumnNumber):
    global gateway
    if (gateway == None):
        gateway = JavaGateway()

    for index in range(len(csvFileNames)):
        message = gateway.createArffFromCsv(csvFileNames[index], arffFileNames[index], classColumnNumber)
        util.thisLogger.logInfo(message)


# ----------------------------------------------------------------------------
def getExecutionDirectory():
    message = gateway.getExecutionDirectory()
    util.thisLogger.logInfo(message)
    return message


# ----------------------------------------------------------------------------
def stopProcessing():
    global process
    process = False
    util.thisLogger.logInfo('processing stop request received: process=' + str(process))


# ----------------------------------------------------------------------------
def processStreamInstance(i, clustererName, numInstances, batchProcessUnseenInstances=False, saveOutlierResults=False):
    global results, streamList, clustererList, gatewayDict, numProcessedInst, batchSize
    if saveOutlierResults == True:
        if results == None:
            results = []

    trainFilename = os.path.join(os.path.join(*util.getFolderList(__file__)[:-1]), "output/trainingactivations_%s.csv" % (i))

    numSamples = 1
    dataFound = False
    instIds = []
    newInstances = []
    instanceCount = numInstances
    try:
        while streamList[i].hasMoreInstances():
            newInstEx = streamList[i].nextIdInstance()
            if (newInstEx is not None):
                instId = newInstEx.getIdAsString()
                newInst = newInstEx.getInstanceExample().getData()

                if saveOutlierResults == True:
                    if batchProcessUnseenInstances == False:
                        isInstanceAnOutlier(newInst, instId, numInstances, clustererName, clustererList[i], i, numSamples,
                                            trainFilename)
                    else:
                        newInstances.append(newInst)
                        instIds.append(instId)
                else:
                    # add instance as a training instance and do not do any outlier analysis on it
                    gatewayDict.entry_point.Moa_Clusterers_Outliers_MCOD_processNewInstanceImplTrain(
                        clustererList[i], newInst)
                    msg = 'Received %s instances from stream %s. Applying to clusterer' % (numSamples, i)

                    # print message every 1000 instances
                    if numSamples == instanceCount:
                        util.thisLogger.logInfo(msg)
                        instanceCount = instanceCount + 1000

                dataFound = True
                if saveOutlierResults:  # unseen instance
                    numProcessedInst += 1
                numSamples += 1

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        util.thisLogger.logInfo(
            'problem reading stream: %s, %s, %s, %s, %s' % (e, exc_type, exc_obj, exc_tb, exc_tb.tb_lineno))
        if gateway != None:
            gatewayDict.entry_point.StopCreatingTrainedClusterers()

    if (dataFound):
        if batchProcessUnseenInstances == True:
            isInstancesAnOutlier(newInstances, instIds, clustererList[i], i)
        if numProcessedInst == batchSize:
            stopProcessing()


# ----------------------------------------------------------------------------
def isInstanceAnOutlier(newInst, instId, numInstances, clusterer, i, numSamples, trainFilename):
    global results, gatewayDict
    # determine if the instance is an outlier
    outlierResult = gatewayDict.entry_point.Moa_Clusterers_Outliers_MCOD_addAndAnalyse(clusterer, newInst,
                                                                                                   trainFilename,
                                                                                                   numCpus)

    if (outlierResult[0] == 'DATA,OUTLIER,OUTLIER'):
        util.thisLogger.logInfoColour(
            "[%s] Activation data for stream %s, instance %s: %s" % (instId, i, numSamples, outlierResult),
            'red')
    elif (outlierResult[0] == 'DATA,OUTLIER,NOT_OUTLIER'):
        util.thisLogger.logInfoColour(
            "[%s] Activation data for stream %s, instance %s: %s" % (instId, i, numSamples, outlierResult),
            'green')
    else:
        util.thisLogger.logInfoColour(
            "[%s] Activation data for stream %s, instance %s: %s" % (instId, i, numSamples, outlierResult),
            'magenta')

    result = [instId, i, outlierResult.replace('DATA,OUTLIER,', '').split(',')[1]]
    results.append(result)


# ----------------------------------------------------------------------------
def isInstancesAnOutlier(newInstances, instIds, clusterer, i):
    global results, gatewayDict

    # convert instances to java list
    jNewInstances = ListConverter().convert(newInstances, gatewayDict._gateway_client)
    jInstIds = ListConverter().convert(instIds, gatewayDict._gateway_client)

    outlierResults = gatewayDict.entry_point.Moa_Clusterers_Outliers_MCODSingle_addAndAnalyse(clusterer, jNewInstances, jInstIds)

    for outlierResult in outlierResults:
        instId = outlierResult.split(',')[0]
        if 'DATA,OUTLIER,OUTLIER' in outlierResult:
            util.thisLogger.logInfoColour("Activation data for stream %s: %s" % (i, outlierResult), 'red')
        elif 'DATA,OUTLIER,NOT_OUTLIER' in outlierResult:
            util.thisLogger.logInfoColour("Activation data for stream %s: %s" % (i, outlierResult), 'green')
        else:
            util.thisLogger.logInfoColour("Activation data for stream %s: %s" % (i, outlierResult), 'magenta')

        result = [instId, i, outlierResult.replace('DATA,OUTLIER,', '').split(',')[1]]
        results.append(result)


# ----------------------------------------------------------------------------
def createStream(jActLabels, dataSourceName):
    global gatewayDict
    # set the stream
    stream = None
    if (dataSourceName == 'rbf'):
        stream = gatewayDict.entry_point.Moa_Streams_Generators_RandomRBFGenerator_New()

    if (dataSourceName == 'add'):
        stream = gatewayDict.entry_point.Moa_Streams_AddStream_New(jActLabels)

    stream.prepareForUse()
    return stream


# ----------------------------------------------------------------------------
def createClusterer(stream, dataClass):
    global gatewayDict
    # set the clusterer
    clusterer = gatewayDict.entry_point.Moa_Clusterers_Outliers_MCOD_New()
    k = util.getParameter('mcod_k')[0].item()
    radius = util.getParameter('mcod_radius')[0]
    windowSize = util.getParameter('mcod_windowsize')
    setAnalysisParameters_Mcod(clusterer, k, radius, windowSize, dataClass)

    clusterer.setModelContext(stream.getHeader())
    clusterer.prepareForUse()
    return clusterer

# ----------------------------------------------------------------------------
def batchProcessObjList(unseenDataList, dnnPredict_batch):
    global streamList

    for c in np.unique(dnnPredict_batch).astype(int):
        instances = [u for u,p in zip(unseenDataList, dnnPredict_batch) if p == c]
        ids = [str(uuid.uuid4()) for i in instances]
        streamName = str(c)
        labels = np.arange(len(instances[0]))
        instances = np.concatenate(([labels], instances), axis=0)  # axis=0 means add row
        instancesCsvFile = os.path.join(os.path.join(*util.getFolderList(__file__)[:-1]),"output/unseenactivations_%s.csv" % (streamName))

        util.saveToCsv(instancesCsvFile, instances)

        try:
            addUnseenDataToStreams_MoaFile(streamList[streamName], instancesCsvFile, ids, streamName)
        except:
            # there is no stream setup for prediction.  There were no correct training instances for this class
            # that has been predicted in the unseen instances
            raise Exception(
                'There is no stream setup for prediction %s.  This means that there were no correct training data predictions for prediction %s, so no stream was created' % (
                    streamName, streamName))

        util.deleteFile(instancesCsvFile)

# ----------------------------------------------------------------------------
def getAnalysisParams():
    params = []
    global clustererList, dsData
    dataClass = util.getParameter('DataClasses')[0] # get the first data class

    # get the parameters for the first class to check they have been set
    paramStr = gatewayDict.entry_point.Moa_Clusterers_Outliers_MCOD_GetParameters(clustererList[str(dataClass)])
    paramArry = paramStr.split(',')
    params = [p.split(' = ') for p in paramArry]

    return params








