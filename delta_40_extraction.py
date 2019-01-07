import pickle as cPickle
import numpy as np
from sklearn import preprocessing
import python_speech_features as mfcc
from python_speech_features import delta
from scipy.io.wavfile import read

source   = "/home/deeplearning/Downloads/development_set "

train_file = "/home/deeplearning/Downloads/development_set_enroll.txt"

file_paths = open(train_file,'r')

count = 1

features = np.asarray(())
features.shape
for path in file_paths:
    path = path.strip()
    print(path)

for path in file_paths:
    rate,audio = read(source + '/' + path)
    mfcc_feat = mfcc.mfcc(audio,rate, 0.025, 0.01,20,appendEnergy = True)
    mfcc_feat = preprocessing.scale(mfcc_feat)
    delta = delta(mfcc_feat,20)
    combined = np.hstack((mfcc_feat,delta))
    combined.shape















########################## ML Modelling ######################################

from sklearn.mixture import GaussianMixture

    gmm_model = GaussianMixture(n_components=16).fit(combined)

test_file = "/home/deeplearning/Downloads/development_set_test.txt"

file_paths = open(test_file,'r')

source   = "/home/deeplearning/Downloads/development_set "


for path in file_paths:
    path = path.strip()
    print(path)

sr,audio = read(source + '/' + path)
vector   = mfcc(audio,sr,0.025,0.01,20,appendEnergy=True)
