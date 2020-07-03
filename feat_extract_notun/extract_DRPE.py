#!/usr/bin/python
import tensorflow as tf
from keras.applications.densenet import DenseNet121
from keras.preprocessing import image
from keras.models import load_model
from keras.applications.densenet import preprocess_input
import numpy as np
import pandas as pd
import os
from datetime import datetime

startTime = datetime.now()

def load_model_disk():
    model = DenseNet121(include_top=False, weights='imagenet')
    model.save(r'/storage/projects/ce903/deep_feat_extract/model.h5')
    
    
def feedForward(fname,model):

    img = image.load_img(fname, target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    features = features.flatten()
    print(features)
    
    return features

def extract_store(path_dir):

    feature_vector = []
    names_vector = []
    model = load_model(r"/storage/projects/ce903/deep_feat_extract/model.h5")
    
    for image in os.listdir(path_dir):
        print("Extracting features for modality")
        path = os.path.join(path_dir,image)
        name_image = os.path.splitext(image)[0]
        print(path)
        print(name_image)
        vector = feedForward(path,model)
        feature_vector.append(vector)
        names_vector.append(name_image)

    df_values = pd.DataFrame(feature_vector, dtype = float)
    df_ids = pd.DataFrame(names_vector)
    df_store = pd.concat((df_ids,df_values),axis = 1)
    
    return df_store 

if os.path.isfile(r'/storage/projects/ce903/deep_feat_extract/model.h5'):
    print("Model already in disk")
    folder = 'DRPE'
    path = os.path.join(r'/storage/projects/ce903/Train_images/',folder)
    print(path)
    df = extract_store(path)
    df.to_csv(r'/storage/projects/ce903/deep_feat_extract/Features_%s.csv' %folder, index = False)
else:
    print("Load model :)")
    load_model_disk()
    
print(datetime.now() - startTime)   



  
  