from scipy.spatial import distance
import numpy as np
import pandas as pd
import os
from datetime import datetime
import csv
import multiprocessing
import multiprocessing.pool
from itertools import repeat
from datetime import datetime
startTime = datetime.now()

#Credits to Chris Arndt from stackoverflow because I was able to handle multiprocessing inside a multiprocess
class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess
    

num_cores = multiprocessing.cpu_count() #Count the amount of cores 
num_cores_1 = num_cores - 20 #assign n - 30 cores to one of the multiprocesses


#Takes array of similar images and their content tags, finds the most similar tags and returns them
def tagSelection(tagArray):
    tags = {}
    value = 10
    threshold = 20
    for index, row in tagArray.iterrows():
        temp = row[2].split()
        for i in temp:
            if i in tags:
                tags[i] = tags[i] + value
            else:
                tags[i] = value
        value -= 1
    finalTags = ''
    for i, j in tags.items():
        if j >= threshold:
            finalTags += i + ";"
    finalTags = finalTags[:-1]
    return finalTags
        
#Takes an image along with a database of feature vectors of other images, and finds 3 similarity measurements between the input image and the ones in the database
def compute_sim_can(df,currentVec,simArray):
    
    for index, row in df.iterrows():
        imageName = row[0] + '.jpg'
        canDis = distance.canberra(currentVec, row[1:])
        temp = pd.DataFrame([[imageName, canDis]], columns=['ImageName','Canberra'])
        simArray = simArray.append(temp, ignore_index=True)
    
    return simArray

def compute_sim_bray(df,currentVec,simArray):
    
    for index, row in df.iterrows():
        imageName = row[0] + '.jpg'
        brayDis = distance.braycurtis(currentVec, row[1:])
        temp = pd.DataFrame([[imageName,brayDis]], columns=['ImageName','BrayCurtis'])
        simArray = simArray.append(temp, ignore_index=True)
    
    return simArray
    
def compute_sim_mh(df,currentVec,simArray):
    
    for index, row in df.iterrows():
        imageName = row[0] + '.jpg'
        manDis = distance.cityblock(currentVec, row[1:])
        temp = pd.DataFrame([[imageName,manDis]], columns=['ImageName','Manhatten'])
        simArray = simArray.append(temp, ignore_index=True)
    
    return simArray
    
## for the next three functions, credits to mlee_jordan from stackoverflow. I based the functions based on an example he uploaded
def similarityMeasures_can(image,fname, vecFile, tagFile,canberra):
    currentVec = image
    featureVecs = pd.read_csv(vecFile)
    contentTags = pd.read_csv(tagFile)
    simArray = pd.DataFrame(columns=['ImageName','Canberra'])
    list_df = list()
    for idx,x in enumerate(range(num_cores_1)):
      df = np.array_split(featureVecs,num_cores_1)[idx]
      list_df.append(df)
    
    args = [] 
    
    for idx,x in enumerate(list_df):
      args.append(tuple((list_df[idx],currentVec,simArray)))

      
    with multiprocessing.Pool(processes= num_cores_1) as pool:
        simArray = pd.concat(pool.starmap(compute_sim_can,args))
    
    print("Similarity measurements complete")
    #print(simArray)
    #Most Similar by Canberra Distance
    #simArray.to_csv(r'/storage/projects/ce903/canberra_DRCO_param.csv', index = False)
    can = simArray[['ImageName','Canberra']].copy()
    can = can.sort_values(by=['Canberra'])
    can = can.head(10)
    tagArray = pd.merge(can, contentTags, on='ImageName')
    tags = tagSelection(tagArray)
    temp = pd.DataFrame([[fname, tags]], columns=['ImageName', 'Tags'])
    canberra = canberra.append(temp, ignore_index=True)
    return canberra

def similarityMeasures_bray(image,fname, vecFile, tagFile, brayCurtis):
    currentVec = image
    featureVecs = pd.read_csv(vecFile)
    contentTags = pd.read_csv(tagFile)
    simArray = pd.DataFrame(columns=['ImageName','BrayCurtis'])
    list_df = list()
    for idx,x in enumerate(range(num_cores_1)):
      df = np.array_split(featureVecs,num_cores_1)[idx]
      list_df.append(df)
    
    args = [] 
    
    for idx,x in enumerate(list_df):
      args.append(tuple((list_df[idx],currentVec,simArray)))

      
    with multiprocessing.Pool(processes= num_cores_1) as pool:
        simArray = pd.concat(pool.starmap(compute_sim_bray,args))
    
    print("Similarity measurements complete")
    #print(simArray)
    #Most Similar by Bray Curtis Dissimilarity
    bray = simArray[['ImageName','BrayCurtis']].copy()
    bray = bray.sort_values(by=['BrayCurtis'])
    bray = bray.head(10)
    tagArray = pd.merge(bray, contentTags, on='ImageName')
    tags = tagSelection(tagArray)
    temp = pd.DataFrame([[fname, tags]], columns=['ImageName', 'Tags'])
    brayCurtis = brayCurtis.append(temp, ignore_index=True)
    
    return brayCurtis
 
def similarityMeasures_mh(image,fname, vecFile, tagFile, manhatten):
    currentVec = image
    featureVecs = pd.read_csv(vecFile)
    contentTags = pd.read_csv(tagFile)
    simArray = pd.DataFrame(columns=['ImageName','Manhatten'])
    list_df = list()
    for idx,x in enumerate(range(num_cores_1)):
      df = np.array_split(featureVecs,num_cores_1)[idx]
      list_df.append(df)
    
    args = [] 
    
    for idx,x in enumerate(list_df):
      args.append(tuple((list_df[idx],currentVec,simArray)))

      
    with multiprocessing.Pool(processes= num_cores_1) as pool:
        simArray = pd.concat(pool.starmap(compute_sim_mh,args))
    
    print("Similarity measurements complete")
    #print(simArray)
    #Most Similar by Manhatten Distance
    man = simArray[['ImageName','Manhatten']].copy()
    man = man.sort_values(by=['Manhatten'])
    man = man.head(10)
    tagArray = pd.merge(man, contentTags, on='ImageName')
    tags = tagSelection(tagArray)
    temp = pd.DataFrame([[fname, tags]], columns=['ImageName', 'Tags'])
    manhatten = manhatten.append(temp, ignore_index=True)
    return manhatten   

#model=load_model("/storage/projects/ce903/model.h5") #Load pretrained model

if __name__ == "__main__":
  
  folders = ['DRAN','DRCO','DRCT','DRMR','DRPE','DRUS','DRXR']
  canberra_all = pd.DataFrame(columns=['ImageName','Tags'])
  brayCurtis_all = pd.DataFrame(columns=['ImageName','Tags'])
  manhatten_all = pd.DataFrame(columns=['ImageName','Tags'])
  
  for folder in folders:  
    canberra = pd.DataFrame(columns=['ImageName','Tags'])
    brayCurtis = pd.DataFrame(columns=['ImageName','Tags'])
    manhatten = pd.DataFrame(columns=['ImageName','Tags'])
    csvname = "/storage/projects/ce903/deep_feat_extract/Features_%s.csv" %folder  ##Path to database images (training)
    tagFile = "/storage/projects/ce903/%s_CLEF_Train_Merged.csv" %folder  #Path to csv file with the tags of each image in training image set
    path = os.path.join("/storage/projects/ce903/test_bench_feat/","Features_%s.csv" %folder) #Path to features (testing set) of images to predict
    
    pool_1 = MyPool(20)
    df_val_images = pd.read_csv(path)
    list_val = []
    args_val_can = []
    args_val_bray = []
    args_val_mh = []
    
    for index, rows in df_val_images.iterrows():
        my_list = [rows[0],rows[1:].tolist()]
        list_val.append(my_list)
    
    for idx,x in enumerate(list_val):
          args_val_can.append(tuple((list_val[idx][1],list_val[idx][0],csvname, tagFile,canberra)))
          args_val_bray.append(tuple((list_val[idx][1],list_val[idx][0],csvname, tagFile,brayCurtis)))
          args_val_mh.append(tuple((list_val[idx][1],list_val[idx][0],csvname, tagFile, manhatten)))
     
         
    #for idx,row in enumerate(list_val):
    #
    #    canberra,brayCurtis,manhatten = similarityMeasures(list_val[idx][1],list_val[idx][0],csvname, tagFile,canberra,brayCurtis, manhatten)
    #    print("Done")
    
    #Credits to Chris Arndt from stackoverflow because I was able to handle multiprocessing inside a multiprocess
      
    canberra= pd.concat(pool_1.starmap(similarityMeasures_can,args_val_can))
    canberra_all = canberra_all.append(canberra)
    canberra.to_csv(r'/storage/projects/ce903/cs_test_bench/experiment_1/canberra_%s_0.csv' %folder, index = False)
    
    
    brayCurtis = pd.concat(pool_1.starmap(similarityMeasures_bray,args_val_bray))
    brayCurtis_all = brayCurtis_all.append(brayCurtis)
    brayCurtis.to_csv(r'/storage/projects/ce903/cs_test_bench/experiment_1/braycurtis_%s_0.csv' %folder, index = False)
    
    
    manhatten = pd.concat(pool_1.starmap(similarityMeasures_mh,args_val_mh))
    manhatten_all = manhatten_all.append(manhatten)
    manhatten.to_csv(r'/storage/projects/ce903/cs_test_bench/experiment_1/manhatten_%s_0.csv' %folder, index = False)
  
  canberra_all.to_csv(r'/storage/projects/ce903/cs_test_bench/experiment_1/canberra_all.csv', index = False)  
  brayCurtis_all.to_csv(r'/storage/projects/ce903/cs_test_bench/experiment_1/brayCurtis_all.csv', index = False)  
  manhatten_all.to_csv(r'/storage/projects/ce903/cs_test_bench/experiment_1/manhattan_all.csv', index = False)  
  
  
  print(datetime.now() - startTime)

