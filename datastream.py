# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 10:22:46 2019

@author: id127392
"""
import numpy as np
import random
import util
import dataset

unseenDataList = None
instanceProcessingStartTime = None
numberNdUnseenInstances = 0
numberDUnseenInstances = 0

#----------------------------------------------------------------------------
def getData(isApplyDrift=True):
    # Gets the unseen data from the dataset
    global numberNdUnseenInstances, numberDUnseenInstances
    ndUnseenDataList = []
    dUnseenDataList = []
    numUnseenInstances = util.getParameter('NumUnseenInstances')
    f_x_train, f_y_train, f_x_test, f_y_test = dataset.getFilteredData()
    if util.hasSubClasses():
        _, u_f_y_train, _, u_f_y_test = dataset.getFilteredData(isMap=False) # if data is sub-classed, get unmapped classes

    if numUnseenInstances == -1:
        numberNdUnseenInstances = len(f_x_test)
    oof_x_train, oof_y_train, oof_x_test, oof_y_test = dataset.getOutOfFilterData()
    if util.hasSubClasses():
        _, u_oof_y_train, _, u_oof_y_test = dataset.getOutOfFilterData(isMap=False)

    if numUnseenInstances == -1:
        numberDUnseenInstances = len(oof_x_test)
    dataDiscrepancy, numDiscrepancy, numNonDiscrepancy = getInstanceParameters()

    numUnseenInstances = util.getParameter('NumUnseenInstances')
        
    # get non-discrepancy data
    if numUnseenInstances != -1:
        f_x_test = f_x_test[:numUnseenInstances]
        f_y_test = f_y_test[:numUnseenInstances]
        if util.hasSubClasses():
            u_f_y_test = u_f_y_test[:numUnseenInstances]

    # get discrepancy data
    if numUnseenInstances != -1:
        oof_x_test = oof_x_test[:numUnseenInstances]
        oof_y_test = oof_y_test[:numUnseenInstances]
        if util.hasSubClasses():
            u_oof_y_test = u_oof_y_test[:numUnseenInstances]

    iCount = 1
    for index in range(numNonDiscrepancy):
        nonDiscrepancyInstance = f_x_test[np.array([index])]
        result = f_y_test[index]
        i = UnseenData(instance=nonDiscrepancyInstance, correctResult=result, discrepancyName='ND')
        i.id = str(iCount)
        if util.hasSubClasses():
            i.hasSubClasses = True
            i.subCorrectResult = u_f_y_test[index]
        ndUnseenDataList.append(i)
        iCount += 1
    # collect discrepancy data and insert randomly
    for index in range(numDiscrepancy):
        discrepancyInstance = oof_x_test[np.array([index])]
        result = oof_y_test[index]
        i = UnseenData(instance=discrepancyInstance, correctResult=result, discrepancyName=util.getParameter('DataDiscrepancy'))
        i.id = str(iCount)
        if util.hasSubClasses():
            i.hasSubClasses = True
            i.subCorrectResult = u_oof_y_test[index]
        dUnseenDataList.append(i)
        iCount += 1

    if isApplyDrift:
        unseenDataList = applyDrift(ndUnseenDataList, dUnseenDataList)
    else:
        # if this method is being called from the getDataFromAllInstFile function, we do not want to apply drift
        ndUnseenDataList.extend(dUnseenDataList)
        unseenDataList = ndUnseenDataList

    return unseenDataList

#----------------------------------------------------------------------------
def EqualizeNumInstances(ndUnseenDataList, dUnseenDataList):
    # make the number of ND and D instances equal so drift can be applied without re-using instances
    numUnseenInstances = util.getParameter('NumUnseenInstances')
    if numUnseenInstances != -1:
        ndUnseenDataList = ndUnseenDataList[:numUnseenInstances // 2]
        dUnseenDataList = dUnseenDataList[:numUnseenInstances // 2]

    numberNdUnseenInstances = len(ndUnseenDataList)
    numberDUnseenInstances = len(dUnseenDataList)

    if numberDUnseenInstances == 0:
        numInst = numberNdUnseenInstances
    else:
        numInst = min(numberNdUnseenInstances, numberDUnseenInstances)

    # take the samples in order from the begining
    ndUnseenDataList = ndUnseenDataList[:numInst]
    dUnseenDataList = dUnseenDataList[:numInst]

    return ndUnseenDataList, dUnseenDataList

#----------------------------------------------------------------------------
def applyDrift(ndUnseenDataList, dUnseenDataList):
    # abrupt, gradual, incremental, recurring, outlier
    unseenDataList = []

    random.shuffle(ndUnseenDataList)
    random.shuffle(dUnseenDataList)
    ndUnseenDataList, dUnseenDataList = EqualizeNumInstances(ndUnseenDataList, dUnseenDataList)
    unseenDataList.extend(ndUnseenDataList)
    unseenDataList.extend(dUnseenDataList)
    random.shuffle(unseenDataList)

    # number the list
    newUnseenDataList = []
    i = 0
    for u in unseenDataList:
        d = UnseenData(id=str(i), instance=u.instance, reducedInstance=u.reducedInstance, correctResult=u.correctResult, discrepancyName=u.discrepancyName,
                       predictedResult=u.predictedResult)
        d.subCorrectResult = u.subCorrectResult
        d.hasSubClasses = u.hasSubClasses
        newUnseenDataList.append(d)
        i += 1

    return newUnseenDataList

#----------------------------------------------------------------------------
class UnseenData:
    def __init__(self, id='', instance=None, reducedInstance=np.asarray([None]), correctResult=None, discrepancyName=None, predictedResult=0):
        self.id = id
        self.instance = instance
        self.reducedInstance = reducedInstance
        self.correctResult = correctResult
        self.discrepancyName = discrepancyName
        self.discrepancyResult = ''
        self.predictedResult = predictedResult
        self.subCorrectResult = np.asarray([None])
        self.hasSubClasses = False
     
        
#----------------------------------------------------------------------------      
def getInstanceParameters():
    global numberNdUnseenInstances, numberDUnseenInstances
    numUnseenInstances = util.getParameter('NumUnseenInstances')

    util.thisLogger.logInfo('NumUnseenInstances=%s' % (numUnseenInstances))

    if numUnseenInstances == -1:
        totalDiscrepancy = numberDUnseenInstances
        totalNonDiscrepancy = numberNdUnseenInstances
    else:
        dataDiscrepancyFrequency = '1in2'
        splitData = dataDiscrepancyFrequency.split('in')
        numDiscrepancy = int(splitData[0].strip())  # first number is number of discrepancies
        numNonDiscrepancy = int(splitData[1].strip()) # second number is number of  non-discrepancies
        ratio = numUnseenInstances/numNonDiscrepancy
        totalDiscrepancy = int(numDiscrepancy*ratio)
        totalNonDiscrepancy = int((numNonDiscrepancy-numDiscrepancy)*ratio)

    dataDiscrepancy = util.getParameter('DataDiscrepancy')
      
    return dataDiscrepancy, totalDiscrepancy, totalNonDiscrepancy
