from openmax import *
from openmax_utils import *
import os

model = None
x_train = None
y_train = None
classes = None

def setupAnalysis(dnn, param_x_train_batch, param_y_train_batch):
    # batch from the stream is provided to this function
    global model, x_train, y_train, classes

    classes = util.getParameter('DataClasses')
    if util.hasSubClasses():
        classes = np.unique(util.getParameter('MapNewYValues'))

    model = dnn
    x_train = param_x_train_batch
    y_train = param_y_train_batch.flatten()

    print('Creating mean activation vector (MAV) and Weibull fit model for each class...')
    # Create mean activation vector (MAV) and weibull fit model
    # This function will create numpy files saving the MAV for future use
    # mean.npy and distance.npy are created by this method
    # delete these files before running this method to ensure old files are not used
    if os.path.exists("mean.npy"):
        os.remove("mean.npy")
    if os.path.exists("distance.npy"):
        os.remove("distance.npy")

    create_model(model, param_x_train_batch, param_y_train_batch, classes)

    print('Model created')

#------------------------------------------------------------------------------------------------
def processUnseenStreamBatch(x_unseen_batch, dnnPredict_batch):
    # batch from the stream is provided to this function
    global model, classes

    displayFullResults = True
    getActivationsByBatch = True
    dataList = x_unseen_batch
    result = []
    alpha_ranks = [2]
    tail_sizes = [9]

    for a in alpha_ranks:
        ALPHA_RANK = a
        for t in tail_sizes:
            TAIL_SIZE = t
            count = 0

            if getActivationsByBatch:
                # compute activations for all images in batch
                activations = compute_activation(model, dataList)
                for scores,fc8 in zip(activations['scores'], activations['fc8']):
                    # Compute openmax
                    activation = {}
                    scores = np.reshape(scores, (1, len(scores)))
                    activation['scores'] = scores
                    fc8 = np.reshape(fc8, (1, len(fc8)))
                    activation['fc8'] = fc8
                    softmax, openmax = compute_openmax(model, activation, ALPHA_RANK, TAIL_SIZE, classes)
                    result = getResult(softmax, openmax, count, dataList, dnnPredict_batch, result, displayFullResults)
                    count += 1

            else:
                for i in dataList:
                    # Compute fc8 activation for the given image
                    i = np.reshape(i, (1, 32, 32, 3))
                    activation = compute_activation(model, i)

                    # Compute openmax
                    softmax, openmax = compute_openmax(model, activation, ALPHA_RANK, TAIL_SIZE, classes)

                    result = getResult(softmax, openmax, count, dataList, dnnPredict_batch, result, displayFullResults)
                    count += 1

    return result # list of 'N','D' or 'U's

def endOfUnseenStream():
    if os.path.exists("mean.npy"):
        os.remove("mean.npy")
    if os.path.exists("distance.npy"):
        os.remove("distance.npy")



def getResult(softmax, openmax, count, dataList, dnnPredict_batch, result, displayFullResults):
    if openmax == -1 or openmax == 2:
        openmax = 'unknown'

    if openmax == 'unknown':
        result.append('D')
    else:
        result.append('N')

    if displayFullResults:
        util.thisLogger.logInfo('DNN: %s  Softmax: %s  Openmax: %s' % (dnnPredict_batch[count], softmax, openmax))
    else:
        if count % 50 == 0:
            print('processing instance %s of %s' % (count, len(dataList)))
    return result

#------------------------------------------------------------------------------------------------
def getAnalysisParams():
    parameters = []
    # return a list of variable parameters in the format [[name,value],[name,value]...[name,value]]
    parameters.append(['none',0])
    return parameters




