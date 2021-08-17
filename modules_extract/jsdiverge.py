import util
import numpy as np
from scipy.spatial import distance
lastLayer = np.asarray([None])

def extractSingleLayer(key, layer, reset=False, layerActivationReduction='jsdiverge', model=None):
    # claculates jensen shannon divergence/distance per layer
    # get js divergence between layers
    global lastLayer
    if reset:
        lastLayer = lastLayer = np.asarray([None])

    result = np.asarray([None])
    if lastLayer.any() == None:
        # store the layer for future use
        lastLayer = layer
    else:
        value1 = lastLayer
        value2 = layer

        jsarray = np.ndarray([np.asarray(value2).shape[0], 1])
        for i in range(np.asarray(value2).shape[0]):
            # for each instance
            val1 = value1[i]
            val1 = np.absolute(val1)
            val2 = value2[i]
            val2 = np.absolute(val2)
            # scipy provides the jensen shannon distance, not the divergence
            js = distance.jensenshannon(val1, val2)

            if layerActivationReduction == 'jsdiverge':
                # square the jensen shannon distance to get the jensen shannon divergence
                js = js * js

                js = np.nan_to_num(js, nan=0)
                if np.isnan(js):
                    util.thisLogger.logInfo('nan found')
                jsarray[i][0] = js

        result = jsarray
        lastLayer = value2

    return result