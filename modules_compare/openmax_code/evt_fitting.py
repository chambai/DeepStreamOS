from openmax_utils import *

try:
    import libmr
except ImportError:
    print ("LibMR not installed or libmr.so not found")
    print ("Install libmr: cd libMR/; ./compile.sh")

#---------------------------------------------------------------------------------
NCHANNELS = 1

#---------------------------------------------------------------------------------
def weibull_tailfitting(mean, distance, tailsize = 10, distance_type = 'eucos'):
                        
    """ Read through distance files, mean vector and fit weibull model for each category

    Input:
    --------------------------------
    meanfiles_path : contains path to files with pre-computed mean-activation vector
    distancefiles_path : contains path to files with pre-computed distances for images from MAV
    labellist : ImageNet 2012 labellist

    Output:
    --------------------------------
    weibull_model : Perform EVT based analysis using tails of distances and save
                    weibull model parameters for re-adjusting softmax scores    
    """
    
    weibull_model = {}
    # for each category, read meanfile, distance file, and perform weibull fitting
    distance_scores = np.array(distance[distance_type])
    meantrain_vec = np.array(mean)

    weibull_model['distances_%s'%distance_type] = distance_scores
    weibull_model['mean_vec'] = meantrain_vec
    weibull_model['weibull_model'] = []
    mr = libmr.MR()
    tailtofit = sorted(distance_scores)[-tailsize:]
    mr.fit_high(tailtofit, len(tailtofit))
    weibull_model['weibull_model'] += [mr]

    return weibull_model

#---------------------------------------------------------------------------------
def query_weibull(category_name, weibull_model, distance_type = 'eucos'):
    """ Query through dictionary for Weibull model.
    Return in the order: [mean_vec, distances, weibull_model]
    
    Input:
    ------------------------------
    category_name : name of ImageNet category in WNET format. E.g. n01440764
    weibull_model: dictonary of weibull models for 
    """
    category_weibull = []
    category_weibull += [weibull_model[category_name]['mean_vec']]
    category_weibull += [weibull_model[category_name]['distances_%s' %distance_type]]
    category_weibull += [weibull_model[category_name]['weibull_model']]

    return category_weibull    

