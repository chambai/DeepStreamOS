# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 12:50:23 2019

@author: id127392
"""
import sys

import numpy as np
import logging
import seaborn as sns
import csv
import datetime
import os
import tensorflow as tf
import platform
import subprocess
import os.path, time

paramDefFile = 'input/paramdef.txt'
setupFile = None
thisLogger = None
isLoggingEnabled = True
logPath = 'output/log.log'
logLevel = 'INFO'
params = None
usedParams = []
sns.set(color_codes=True)

#------------------------------------------------------------------------
def getAllParameters():
    global params
    if(params == None):
        # load param def files
        params = getParamData(paramDefFile, setupFile)

# ------------------------------------------------------------------------
def getParamData(paramDefFile, setupFile):
    params = {}
    paramValue = None
    paramdef = getFileLines(paramDefFile)

    # get param names from def file
    paramNames = []
    for param in paramdef:
        splitParams = param.split("#")
        paramName = splitParams[0].strip()
        paramNames.append(paramName)

    # get param values from setup file
    setup = getFileLines(setupFile)
    paramValues = {}
    setup = [s for s in setup if "#" not in s]
    for paramName in paramNames:
        for param in setup:
            splitParams = param.split("=")
            setupName = splitParams[0].strip()
            if paramName == setupName:
                paramValues[paramName] = splitParams[1].strip()

    # store params in dictionary
    for param in paramdef:
        splitParams = param.split("#")
        if len(splitParams) == 2:
            paramType = 'string'
            paramComment = splitParams[1].strip()
        else:
            paramType = splitParams[1].strip()
            paramComment = splitParams[2].strip()

        # check that all parameters that are defined in param def file are in the setup file
        checkAllParamsInSetupFile = True
        if checkAllParamsInSetupFile:
            paramName = splitParams[0].strip()
            if paramName in paramValues:
                paramValue = paramValues[paramName]
            else:
                raise ValueError('Variable: ' + paramName + ' is defined in param def but does not exist in the setup file: ' +  setupFile)
        else:
            paramValue = paramValues[paramName]

        params[paramName] = (paramType, paramComment, paramValue)

    return params
                    
#------------------------------------------------------------------------       
def getParameter(paramName):
    global usedParams
    global params

    # get parameter values
    getAllParameters()
           
    # get parameter type and value
    paramType = params[paramName][0]
    if paramType == 'bool':
        paramValue = stringToBool(params[paramName][2])
    elif paramType == 'float':
        paramValue = float(params[paramName][2])
    elif paramType == 'int':
        paramValue = int(params[paramName][2])
    elif paramType == 'string':
        paramValue = params[paramName][2]
    elif paramType == 'stringarray':
        paramValue = []
        splitParams = params[paramName][2].split(',')
        for p in splitParams:
            paramValue.append(p.strip())
    elif paramType == 'intarray':
        if params[paramName][2] == '[]':
            paramValue = []
        else:
            paramValue = np.asarray(params[paramName][2].replace('[','').replace(']','').split(',')).astype(int)
    elif paramType == 'floatarray':
        if params[paramName][2] == '[]':
            paramValue = []
        else:
            paramValue = np.asarray(params[paramName][2].replace('[','').replace(']','').split(',')).astype(float)
    else:
        # The value must be one of the values specified
        allowedValues = paramType.split(',')
        allowedValues = [x.strip() for x in allowedValues] # List comprehension
        paramValue = params[paramName][2]
        if paramValue not in allowedValues:
            print(paramValue + ' is not in the list of allowed values for parameter ' + paramName )
            print('Allowed values are: ' + paramType )

    p = '%s=%s' % (paramName, str(paramValue))
    if not any(p in u for u in usedParams):
        usedParams.append('%s=%s' % (paramName,str(paramValue)))

    return paramValue

# ------------------------------------------------------------------------
def setParameter(paramName, paramValue):
    global setupFile
    lines = getFileLines(setupFile)
    for n,l in enumerate(lines):
        splitLine = l.split('=')
        if splitLine[0] == paramName:
            lines[n] = '%s=%s\n'%(splitLine[0],paramValue)
            break;
    saveFileLines(setupFile,lines,'w')

#------------------------------------------------------------------------
def stringToBool(string):
    if string == 'True':
        return True
    elif string == 'False':
        return False
    else:
        raise ValueError

#------------------------------------------------------------------------
def getFileLines(filename):
    file = open(filename, "r")
    lines = file.readlines()
    return lines

#------------------------------------------------------------------------
def saveFileLines(filename, lines, mode = 'x'):
    with open(filename, mode) as file:
        file.writelines(lines)


# ------------------------------------------------------------------------
class Logger:
    def __init__(self):  # double underscores means the function is private
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        global logLevel
        if (logLevel == 'INFO'):
            logging.basicConfig(filename=logPath, level=logging.INFO)
        else:
            logging.basicConfig(filename=logPath, level=logging.DEBUG)

    # class methods always take a class instance as the first parameter
    def logInfo(self, item):
        global logLevel
        item = prefixDateTime(item)
        if (logLevel == 'INFO'):
            print(item)
            # print("\x1b[31m\"red\"\x1b[0m")
        global isLoggingEnabled
        if (isLoggingEnabled == True):
            logging.info(item)

    def logInfoColour(self, item, colour):
        global logLevel
        item = prefixDateTime(item)
        if (logLevel == 'INFO'):
            print(getColourText(item, colour))
            # print("\x1b[31m\"red\"\x1b[0m")
        global isLoggingEnabled
        if (isLoggingEnabled == True):
            logging.info(item)

    def logDebug(self, item):
        global logLevel
        item = prefixDateTime(item)
        if (logLevel == 'DEBUG'):
            print(item)
        global isLoggingEnabled
        if (isLoggingEnabled == True):
            logging.debug(item)

    def logError(self, item):
        item = prefixDateTime(item)
        print(item)
        global isLoggingEnabled
        if (isLoggingEnabled == True):
            logging.error(item)

    def closeLog(self):
        for handler in logging.root.handlers[:]:
            handler.close()
            logging.root.removeHandler(handler)


# ------------------------------------------------------------------------
def getColourText(text, colour):
    printCode = None
    if (colour == 'red'):
        printCode = "\x1b[31m" + text + "\x1b[0m"
    elif (colour == 'green'):
        printCode = "\x1b[32m" + text + "\x1b[0m"
    elif (colour == 'blue'):
        printCode = "\x1b[34m" + text + "\x1b[0m"
    elif (colour == 'magenta'):
        printCode = "\x1b[35m" + text + "\x1b[0m"
    elif (colour == 'cyan'):
        printCode = "\x1b[36m" + text + "\x1b[0m"
    elif (colour == 'black'):
        printCode = "\x1b[30m" + text + "\x1b[0m"
    else:
        raise ValueError('Colour: ' + colour + ' is not a recognised colour')

    return printCode

#------------------------------------------------------------------------
def prefixDateTime(item):
    if item:
        item = '%s %s'%(datetime.datetime.now(),item)
    return item

#------------------------------------------------------------------------
def filterDataByClass(x_data, y_data, class_array):
    ix = np.isin(y_data, class_array)
    ixArry = np.where(ix)
    indexes = ixArry[0]
    x_data = x_data[indexes]
    y_data = y_data[indexes]
    return x_data, y_data

#------------------------------------------------------------------------
def transformZeroIndexDataIntoClasses(y_data, classes):
    # changes y data that is from 0 to n into the classes
    # Elements in y_data that are 0 will be changed to the first element in the classes list,
    # elements in y_data that are 1 will be changed to the second element in the classes list etc...
    y_data_new = np.copy(y_data)
    count = 0
    for c in classes:
        for i, y in enumerate(y_data):
            if y == count:
                y_data_new[i] = c
        count += 1
    y_data = y_data_new
    del y_data_new
    return y_data

#------------------------------------------------------------------------
def transformDataIntoZeroIndexClasses(y_data, originalClasses = [], newClasses=[]):
    # changes y data that is from 0 to n into the classes
    # Elements in y_data that are 0 will be changed to the first element in the classes list,
    # elements in y_data that are 1 will be changed to the second element in the classes list etc...
    y_data_new = np.copy(y_data)
    count = 0
    for c in originalClasses:
        for i, y in enumerate(y_data):
            if y == c:
                if len(newClasses) > 0:
                    y_data_new[i] = newClasses[originalClasses.tolist().index(c)]
                else:
                    y_data_new[i] = count   # no classes specified, assign classes sequentially from zero
        count += 1
    y_data = y_data_new
    del y_data_new
    return y_data

#------------------------------------------------------------------------
def chunks(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i+n]

#------------------------------------------------------------------------
def saveReducedData(filename, flatActivations, y_train):
    classes = np.unique(y_train)

    # create a csv file for each class
    csvFileNames = []
    for dataClass in classes:
        csvFileName = "%s_%s.csv" % (filename, dataClass)
        filteredActivations = []
        for index in range(len(flatActivations)):
            if (y_train[index] == dataClass):
                filteredActivations.append(flatActivations[index])

        if len(filteredActivations) > 0:
            labels = np.arange(len(filteredActivations[0]))
            filteredActivations = np.concatenate(([labels], filteredActivations), axis=0)  # axis=0 means add rows
            thisLogger.logInfo("saving reduced activations to csv file %s" % (csvFileName))
            saveToCsv(csvFileName, filteredActivations)
            csvFileNames.append(csvFileName)

# ------------------------------------------------------------------------
# saves a vector to a csv file
def saveToCsv(csvFilePath, vector, append=False):
    fileWriteMethod = 'w'
    if append == True:
        fileWriteMethod = 'a'

    with open(csvFilePath, fileWriteMethod, newline='') as csv_file:
        csvWriter = csv.writer(csv_file, delimiter=',')
        csvWriter.writerows(vector)

#------------------------------------------------------------------------
def killMoaGateway():
    isWin = False
    info = getSystemInfo()
    if 'Windows' in info:
        isWin = True
    if isWin:
        import wmi
        f = wmi.WMI()
        for process in f.Win32_Process():
            if (process.commandLine != None and 'java.exe' in process.name and 'moagateway' in process.commandLine) or (process.commandLine != None and 'cmd.exe' in process.name and 'subprocess\\moagateway' in process.commandLine):
                thisLogger.logInfo('Terminating Process: %s, %s' % (process.name, process.commandLine))
                process.Terminate()
    else:
        stream = os.popen('ps aux | grep -i moagateway')
        output = stream.read()
        if 'MoaGateway' in output:
            print('yes')
            splitoutput = output.strip().split('\n')
            print(splitoutput)
            for line in splitoutput:
                # remove multiple whitespaces, split on whitespace and get the second element, which is the process ID
                pid = " ".join(line.split()).split(' ')[1]
                print('killing process: ' + pid)
                stream = os.popen('sudo kill ' + pid)
                stream = os.popen('ps aux | grep -i moagateway')
                output = stream.read()
                print(output)
                time.sleep(3)

#------------------------------------------------------------------------
def startMoaGateway(moagatewaybatfile):
    success = False
    output = ''
    isWin = False
    info = getSystemInfo()
    if 'Windows' in info:
        isWin = True

    if isWin:
        # start process
        import wmi
        f = wmi.WMI()
        process_startup = f.Win32_ProcessStartup.new()
        process_startup.ShowWindow = 1
        process_id, result = f.Win32_Process.Create(
            CommandLine='C:\\WINDOWS\\system32\\cmd.exe /c ""' + moagatewaybatfile + '" "',
            ProcessStartupInformation=process_startup
        )
        if result == 0:
            thisLogger.logInfo('Process started successfully: %d' % (process_id))
            success = True
        else:
            raise RuntimeError("Problem creating process: %d" % result)
    else:
        cmd = 'sudo java -Xmx393216m -jar subprocess/MoaGateway.jar'
        stream = os.popen(cmd)
        # output never comes back as it is waiting for the program to end
        stream = os.popen('ps aux | grep -i moagateway')
        output = stream.read()
        if 'MoaGateway' in output:
            success = True
        print('Started MoaGateway: %s'%(success))
        stream = os.popen('ps aux | grep -i moagateway')
        output = stream.read()
        time.sleep(3)

    print(output)
    return success


# ------------------------------------------------------------------------
def getSystemInfo():
    sysInfo = ''
    # get computer info
    uname = platform.uname()
    sysInfo += 'System:%s;' % (uname.system)
    sysInfo += 'Node Name:%s;' % (uname.node)
    sysInfo += 'Release:%s;' % (uname.release)
    sysInfo += 'Version:%s;' % (uname.version)
    sysInfo += 'Machine:%s;' % (uname.machine)
    sysInfo += 'Processor:%s;' % (uname.processor)

    try:
        sp = subprocess.Popen(['nvidia-smi', '-q'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        out_str = sp.communicate()
        out_list = str(out_str[0]).split('\\n')

        out_dict = {}

        for item in out_list:
            if 'Product Name' in item:
                product = item.replace('Product Name', 'GPU')
                sysInfo += product.replace(' ', '') + ';'
            if 'CUDA Version' in item:
                sysInfo += item.replace(' ', '') + ';'

        sysInfo += 'NumGPU:%s' % (len(tf.config.experimental.list_physical_devices('GPU')))

    except:
        # GPU information could not be retrieved
        sysInfo += 'NumGPU:ERROR:%s' % (sys.exc_info()[1])

    return sysInfo

#------------------------------------------------------------------------
def getFolderList(fullFileName):
    drive, path_and_file = os.path.splitdrive(fullFileName)
    path, file = os.path.split(path_and_file)

    folders = []
    while 1:
        path, folder = os.path.split(path)

        if folder != "":
            folders.append(folder)
        elif path != "":
            folders.append(path)

            break

    folders.append(drive)
    folders.reverse()

    return folders

#----------------------------------------------------------------------------
def hasSubClasses():
    mapNewYValues = getParameter('MapNewYValues')
    if len(mapNewYValues) == 0:
        return False
    else:
        return True

#------------------------------------------------------------------------
# delete file
def deleteFile(fullFileName):
    try:
        os.remove(fullFileName)
    except Exception as e:
        print("Haven't deleted file %s (error: %s)" % (fullFileName, e))