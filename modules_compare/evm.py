# EVM is from https://pypi.org/project/EVM/
import EVM, numpy as np, scipy
import util

key = None
classes = None
mevms = {}
numCores = None
probabilityThreshold = None

def setupAnalysis(dnn, x_train_batch, y_train_batch):
    # batch from the stream is provided to this function
    global key, classes, mevms, numCores, probabilityThreshold

    y_train_batch = y_train_batch.flatten()
    xshape = x_train_batch.shape
    x_train_batch = np.reshape(x_train_batch, (xshape[0], xshape[1]*xshape[2]*xshape[3]))
    x_train_batch = x_train_batch.astype('double')

    util.thisLogger.logInfo('Splitting data into classes')
    # get x values for each class
    cBatches = []
    zipBatch = list(zip(x_train_batch, y_train_batch))
    classes = np.sort(util.getParameter('DataClasses'))
    dnn = util.getParameter('Dnn')
    datasetName = util.getParameter('DatasetName')
    discrepancy = util.getParameter('DataDiscrepancy')
    key = '%s-%s-%s-'%(dnn,datasetName,discrepancy) + "-".join(str(x) for x in classes)

    if util.hasSubClasses():
        sourceClasses = util.getParameter('MapOriginalYValues')
        targetClasses = util.getParameter('MapNewYValues')
        subClasses = util.transformDataIntoZeroIndexClasses(classes,sourceClasses,targetClasses)
    else:
        subClasses = classes

    numCores = None
    util.thisLogger.logInfo('Processing on %s cores' % (numCores))
    tailsize = 60
    cover_threshold = 1.0
    probabilityThreshold = 0.5
    util.thisLogger.logInfo('tailsize=%s' % (tailsize))
    util.thisLogger.logInfo('cover_threshold=%s' % (cover_threshold))
    util.thisLogger.logInfo('parallel=%s' % (numCores))
    util.thisLogger.logInfo('probabilityThreshold=%s' % (probabilityThreshold))
    distance_function = scipy.spatial.distance.cosine
    util.thisLogger.logInfo('distance_function=%s' % (str(distance_function).split(' ')[1]))

    if key not in mevms:
        for c in subClasses:
            cBatch = [z[0] for z in zipBatch if z[1] == c]
            cBatches.append(cBatch)

        # For a given set of samples of several classes, it will compute an EVM model for each of the classes, taking all other classes as negatives.
        mevm = EVM.MultipleEVM(tailsize=tailsize, cover_threshold=cover_threshold,
                               distance_function=distance_function)

        util.thisLogger.logInfo('Fitting the training set on %s, %s'%(key, discrepancy))
        mevm.train(cBatches, parallel=numCores)
        util.thisLogger.logInfo('Fitted the training set')
        mevms[key] = mevm
    else:
        util.thisLogger.logInfo('EVM for classes %s already exists' % (classes))



def processUnseenStreamBatch(x_unseen_batch, dnnPredict_batch):
    # do nothing here - return result of NDs
    global key, classes, mevms, numCores, probabilityThreshold
    x_unseen_batch = x_unseen_batch.astype('double')

    util.thisLogger.logInfo('Processing instances %s  %s' % (key,util.getParameter('DataDiscrepancyClass')))

    xshape = x_unseen_batch[0].shape
    xbatch = np.reshape(x_unseen_batch, (x_unseen_batch.shape[0], xshape[0] * xshape[1] * xshape[2]))

    mevm = mevms[key]
    probabilities, indexes = mevm.max_probabilities(xbatch, parallel=numCores)

    result = []
    for u, pred, i, prob in zip(x_unseen_batch, dnnPredict_batch, indexes, probabilities):
        if prob > probabilityThreshold:  # if the probability of belonging to a class is > 0.5, say this is known.
            res = 'N'
        else:
            res = 'D'
        result.append(res)
        util.thisLogger.logInfo('%s: DNN predicted class: %s, probability: %s' % (res, classes[i[0]], prob))
    return result

def getAnalysisParams():
    parameters = []
    # return a list of variable parameters in the format [[name,value],[name,value]...[name,value]]
    parameters.append(['none',0])
    return parameters