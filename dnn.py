from tensorflow.keras.models import model_from_json

def loadDnn(baseFilename):

    # load json and create model
    jsonFilename = '%s.json'%(baseFilename)
    json_file = open(jsonFilename, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

    # load weights into new model
    weightsFilename = '%s.h5' % (baseFilename)
    model.load_weights(weightsFilename)
    print(model.summary())

    return model