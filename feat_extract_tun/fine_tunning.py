#!/usr/bin/python


#Create by Francisco Parrilla
#This code is to fine-tune the DenseNet-121 model based on the images of a certain modality

#Import libraries
import os
from keras.applications.densenet import DenseNet121
from keras.preprocessing import image
from keras.models import load_model
from keras.applications.densenet import preprocess_input
from keras.layers import GlobalAveragePooling2D, Dense, Flatten
from keras.models import Model
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
import numpy as np
import pandas as pd
import csv


def extract_concepts_train(name):
  concepts = []
  base_dir = r'/storage/projects/ce903/Train_concepts/'
  path = os.path.join(base_dir,name+"_CLEF_Train.csv")
  with open(path, "r") as f:
    reader = csv.reader(f, delimiter=",")
    for i, line in enumerate(reader):
        concept_image = []
        for idx, concept in enumerate(line):
          if idx != 0:
            concept_image.append(concept)
        concepts.append(concept_image)
  return concepts
  
def extract_classes(name):
  concepts = []
  base_dir = r'/storage/projects/ce903/Train_concepts/'
  path = os.path.join(base_dir,name+"_CLEF_Train.csv")
  with open(path, "r") as f:
    reader = csv.reader(f, delimiter=",")
    for i, line in enumerate(reader):
        for idx, concept in enumerate(line):
          if idx != 0:
            concepts.append(concept)
  return concepts
  
def extract_img_path(name):
  concepts = []
  base_dir = r'/storage/projects/ce903/Train_concepts/'
  path = os.path.join(base_dir,name+"_CLEF_Train.csv")
  with open(path, "r") as f:
    reader = csv.reader(f, delimiter=",")
    for i, line in enumerate(reader):
        for idx, concept in enumerate(line):
          if idx == 0:
            concepts.append(concept)
  return concepts
  
def create_model(base_model,modality):
  
  #Pre-processing part
  
  concepts = extract_concepts_train(modality)
  imgs_path = extract_img_path(modality)
  classes_concepts = set(extract_classes(modality))
  
  df = pd.DataFrame({'id_images': imgs_path})
  df['labels'] = pd.Series(concepts)
  
  #Fine tunning part
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  x = Dense(64, activation='relu')(x)
  x = Dropout(0.4)(x)
  predictions = Dense(len(classes_concepts), activation='sigmoid')(x)
  model = Model(inputs=base_model.input, outputs=predictions)

  for layer in base_model.layers:
          layer.trainable = False
          
  model.compile(loss='categorical_crossentropy', 
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])
  
  train_datagen = ImageDataGenerator(validation_split = 0.2 ,rescale=1./255)
  
  train_generator = train_datagen.flow_from_dataframe(df, directory=r"/storage/projects/ce903/Train_images/%s" %modality,
                                                    x_col = "id_images",y_col = "labels",
                                                    batch_size = 32, seed = 1, shuffle = True, subset = "training",
                                                    class_mode = "categorical",classes = classes_concepts,target_size=(64,64))

  validation_generator = train_datagen.flow_from_dataframe(df, directory=r"/storage/projects/ce903/Train_images/%s" %modality,
                                                    x_col = "id_images",y_col = "labels",
                                                    batch_size = 32, seed = 1, shuffle = True, subset = "validation",
                                                    class_mode = "categorical",classes = classes_concepts,target_size=(64,64))
  
  #print(train_generator.classes)
  history = model.fit_generator(train_generator,
                              steps_per_epoch=100,
                              epochs=10,
                              validation_data=validation_generator,
                              validation_steps=10)
  
  for layer in model.layers[:400]:
    layer.trainable = False
  for layer in model.layers[400:]:
    layer.trainable = True
  
  history = model.fit_generator(train_generator,
                              steps_per_epoch=100,
                              epochs=10,
                              validation_data=validation_generator,
                              validation_steps=10)
  model.save('fine-tuned-%s.h5' %modality)
  
if __name__ == "__main__":
  
  modalities = ["DRAN","DRCT","DRMR","DRXR","DRPE","DRCO","DRUS"]
  
  base_model = DenseNet121(include_top=False, weights='imagenet')
  
  for modality in modalities:
    create_model(base_model,modality)
  


