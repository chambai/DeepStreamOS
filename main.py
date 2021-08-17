import util, dnn, dataset, train, analyse, datastream
import numpy as np

def start(dnnName, datasetName, dataType, dataCombination, analysis):
    if __name__ == '__main__':
        trainedClasses = dataCombination.split('-')[0]
        unknownClasses = dataCombination.split('-')[1]

        # load setup file
        util.params = None
        util.usedParams = []
        util.setupFile = 'input/%s_%s_%s_%s.txt'%(dnnName,datasetName,trainedClasses,unknownClasses)
        util.setParameter('Dnn', dnnName)
        util.setParameter('DatasetName', datasetName)
        util.setParameter('DataDiscrepancy', dataType)
        util.setParameter('DataClasses', '[%s]'%(','.join(list(trainedClasses))))
        if analysis == 'mcod':
            util.setParameter('AnalysisModule', 'modules_analysis.%s'%(analysis))
            util.setParameter('LayerActivationReduction', 'flatten,pad,jsdiverge')
        else:
            util.setParameter('AnalysisModule', 'modules_compare.%s' % (analysis))
            util.setParameter('LayerActivationReduction', 'none')
        util.thisLogger = util.Logger()

        # Load the DNN
        baseFilename = 'input/%s_%s_%s_%s' % (dnnName, datasetName, trainedClasses, dataType)
        model = dnn.loadDnn(baseFilename)

        # Get the known class data
        x_train, y_train, x_test, y_test = dataset.getFilteredData()

        if analysis == 'mcod':
            # get activations
            flatActivations = train.getActivations(x_train, -1, model, y_train)
            # normalize
            maxValue = np.amax(flatActivations)
            flatActivations = flatActivations / maxValue
            util.saveReducedData('output/trainingactivations', flatActivations, y_train)
            analyse.loadModule('modules_analysis.' + analysis)
            analyse.setup(flatActivations, y_train)
            unseenData = datastream.getData()
            unseenInstancesObjList = analyse.startDataInputStream(model, False, maxValue, maxValue, unseenData)
        else:
            analyse.loadModule('modules_compare.' + analysis)
            analyse.setupCompare(model, x_train, y_train)
            # pixel data is used and this is already normalized
            # when unseen data is extracted this will be normalized, so we can pass 1 in as the maxValue
            unseenData = datastream.getData()
            unseenInstancesObjList = analyse.startDataInputStream(model, False, 1, 1, unseenData)

        # List the results
        for i in unseenInstancesObjList:
            result = ''
            if i.discrepancyResult == 'D':
                result = 'unknown'
            print('Instance %s: true class: %s, predicted class: %s %s'%(i.id,i.correctResult,i.predictedResult,result))

# Examples - run one at a time
# mobilenet cifar10 class data example (our method)
start(dnnName='mobilenet', datasetName='mnistfashion',dataType='class',dataCombination='012346-57',analysis='mcod')

# mobilenet fashion-MNIST sub-class data example (our method)
# start(dnnName='mobilenet', datasetName='mnistfashion',dataType='sub',dataCombination='05-1237',analysis='mcod')

# mobilenet cifar10 class data example (openmax comparison)
# start(dnnName='mobilenet', datasetName='mnistfashion',dataType='class',dataCombination='012346-57',analysis='openmax')

# mobilenet fashion-MNIST sub-class data example (evm comparison)
# start(dnnName='mobilenet', datasetName='mnistfashion',dataType='sub',dataCombination='05-1237',analysis='evm')




