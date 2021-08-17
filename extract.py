# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 12:04:40 2019

@author: id127392
"""
import numpy as np
from keras import backend as K
import util

lastLayer = np.asarray([None])
module = None

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
    util.thisLogger.logInfo("---------- start of %s activation extraction---------"%(moduleName))
    module.setupAnalysis(flatActivations, y_train)
    util.thisLogger.logInfo("----------- end of %s activaton extraction----------\n"%(moduleName))

#----------------------------------------------------------------------------
def singleLayerActivationReduction(key, layer, reset=False):
    layerActivationReduction = util.getParameter('LayerActivationReduction')
    # if the layer activation reduction method is specified in the test matrix, it will contain dots instead of commas.
    # separate out the dots into elements
    layerActRedArray = []
    for r in layerActivationReduction:
        if '.' in r:
            names = r.split('.')
            for n in names:
                layerActRedArray.append(n)
        else:
            layerActRedArray.append(r)

    for layerReductionName in layerActRedArray:
        util.thisLogger.logInfo('Applying layer activation reduction: %s %s'%(key,layerReductionName))
        if layerReductionName == 'none':
            layerReductionName = 'e_none'

        loadModule('modules_extract.' + layerReductionName)
        layer = module.extractSingleLayer(key, layer, reset=reset)
    return np.asarray(layer)

#-----------------------------------------------------------------------
def getActivations(model, inData):
    global lastLayer
    lastLayer = np.asarray([None])
    layerResults = None
    layers = util.getParameter("IncludedLayers")

    if(layers == 'all'):
        layers = np.arange(len(model.layers))
        # remove first and last elements
        layers = np.delete(layers, [0,len(layers)-1])
    else:
        layers = np.asarray(layers.replace('[','').replace(']','').split(',')).astype(int)

    util.thisLogger.logInfo('Applying activation extraction to layers: %s'%(layers))

    isFirstLayer = True
    for layerNum in layers:
        try:
            getLastLayerOutput = K.function([model.layers[0].get_input_at(0)],
                                      [model.layers[layerNum].output])
        except:
            getLastLayerOutput = K.function([model.layers[0].get_input_at(1)],
                                            [model.layers[layerNum].output])

        if layerNum in layers:
            key = model.layers[layerNum].name

            # reset the global variables in the extract module if it is the first layer
            layerResult = singleLayerActivationReduction(key,getLastLayerOutput([inData])[0], reset=isFirstLayer)
            isFirstLayer = False

            if layerResult.any() == None:
                util.thisLogger.logInfo('did not add layer %s as it was removed due to jsdiverge calculation'%(key))
            elif layerResult is None:
                util.thisLogger.logInfo('did not add layer %s as it is None, as received from the DNN'%(key))
            else:
                if layerResults is None:
                    layerResults = layerResult
                else:
                    layerResults = np.append(layerResults, layerResult, 1)

    numLayers = len(layerResults[0])
    return layerResults, numLayers
