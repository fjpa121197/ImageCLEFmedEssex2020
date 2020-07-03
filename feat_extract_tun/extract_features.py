#!/usr/bin/python

import tensorflow as tf
from keras.applications.densenet import DenseNet121
from keras.preprocessing import image
from keras.models import load_model
from keras.applications.densenet import preprocess_input
from keras import backend as K
import numpy as np
import pandas as pd
import os
from datetime import datetime

startTime = datetime.now()

    
def feedForward(fname,get_layer_output):

    img = image.load_img(fname, target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = get_layer_output([x])[0]
    features = features.flatten()

    return features

def extract_store(path_dir,modality):

    feature_vector = []
    names_vector = []
    model = load_model(r'/storage/projects/ce903/tuned_features/fine-tuned-%s.h5' %modality)
    get_layer_output = K.function([model.layers[0].input],[model.layers[-5].output])
    
    for image in os.listdir(path_dir):
        print("Extracting features for modality")
        path = os.path.join(path_dir,image)
        name_image = os.path.splitext(image)[0]
        print(name_image)
        vector = feedForward(path,get_layer_output)
        feature_vector.append(vector)
        names_vector.append(name_image)

    df_values = pd.DataFrame(feature_vector, dtype = float)
    df_ids = pd.DataFrame(names_vector)
    df_store = pd.concat((df_ids,df_values),axis = 1)
    
    return df_store 

if __name__ == "__main__":

    #modalities = ["DRAN","DRCO","DRCT","DRMR","DRPE","DRUS","DRXR"]
    modalities = ["DRAN"]
    for modality in modalities:
      #path = os.path.join(r'/storage/projects/ce903/Train_images/',modality)
      #df = extract_store(path,modality)
      #df.to_csv(r'/storage/projects/ce903/tuned_features/Train_features/Features_%s.csv' %modality, index = False)
      #path = os.path.join(r'/storage/projects/ce903/Validation_images/',modality)
      #df = extract_store(path,modality)
      #df.to_csv(r'/storage/projects/ce903/tuned_features/Val_features/Features_%s.csv' %modality, index = False)
      path = os.path.join(r'/storage/projects/ce903/test_bench_images/',modality)
      df = extract_store(path,modality)
      df.to_csv(r'/storage/projects/ce903/tuned_features/test_features/Features_%s.csv' %modality, index = False)
      
    
    print(datetime.now() - startTime)   



  
  