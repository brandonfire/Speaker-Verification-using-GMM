import pickle as cPickle
import numpy as np
from sklearn import preprocessing
import python_speech_features as mfcc
from python_speech_features import delta
from scipy.io.wavfile import read
from sklearn.mixture import GaussianMixture
source   = "./Sounddata"

train_file = "./datacollected.txt"

file_paths = open(train_file,'r')

#count = 1

features = np.asarray(())
features.shape

for path in file_paths:
    path = path.strip()
    rate,audio = read(source + '/' + path)
    mfcc_feat = mfcc.mfcc(audio,rate, 0.025, 0.01,20,appendEnergy = True)
    mfcc_feat = preprocessing.scale(mfcc_feat)
    delta = delta(mfcc_feat,20)
    #print("Hello, here is delta,",mfcc_feat)
    combined = np.hstack((mfcc_feat,delta))
    #print(combined)
    gmm_model = GaussianMixture(n_components=16).fit(combined)
    #type(gmm_model)



    #print(gmm_model)









########################## ML Modelling ######################################





test_file = "./datacollected.txt"

file_paths = open(test_file,'r')

#source   = "/home/deeplearning/Downloads/development_set "


for path in file_paths:
    path = path.strip()
    #print(path)

    sr,audio = read(source + '/' + path)
    vector   = mfcc.mfcc(audio,sr,0.025,0.01,20,appendEnergy=True)
#print(path)
#print(vector)