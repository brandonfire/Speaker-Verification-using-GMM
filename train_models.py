import pickle as cPickle
import numpy as np
from scipy.io.wavfile import read
from sklearn.mixture import GaussianMixture
from sklearn import preprocessing
from python_speech_features import delta
from python_speech_features import mfcc
import warnings
import os
warnings.filterwarnings("ignore")

#path to training data
# source   = "/home/deeplearning/Downloads/development_set /"
source = r"./Sounddata"

#path where training speakers will be saved
dest = "./trainedmodel"

train_file = "./datacollected.txt"

file_paths = open(train_file,'r')

count = 1

# Extracting features for each speaker (5 files per speakers)
features = np.asarray(())
features.shape

files = [os.path.join(source,fname) for fname in os.listdir(source) if fname.endswith('.wav')]
print(files)
count = 0
for f in sorted(files):
    print(f)
    if count<5:
        count = count + 1
        sr,audio = read(f)
            # extract 20 dimensional MFCC features
        vector = mfcc(audio,sr,0.025,0.01,20,appendEnergy=True)
        vector.shape
        vector = preprocessing.scale(vector)
        if features.size == 0:
            features = vector
        else:
            features = np.vstack((features, vector))
        print(features)
#print(features.shape)

gmm = GaussianMixture(n_components = 16,covariance_type='diag',n_init = 3)
gmm.fit(features)

        # dumping the trained gaussian model
picklefile = f[11:].split("-")[0]+".gmm"
print(picklefile)
cPickle.dump(gmm,open(dest + picklefile,'wb'))
print('+ modeling completed for speaker:',picklefile," with data point = ",features.shape)
features = np.asarray(())

"""
for path in file_paths:
    path = path.strip()
    print(path)
    # read the audio

sr,audio = read(source + '/' + path)
    # extract 20 dimensional MFCC features
vector = mfcc(audio,sr,0.025,0.01,20,appendEnergy=True)
vector.shape
vector = preprocessing.scale(vector)

    if features.size == 0:
        features = vector
    else:
        features = np.vstack((features, vector))
        features.shape
    # when features of 5 files of speaker are concatenated, then do model training

    if count == 5:
        gmm = GaussianMixture(n_components = 16,covariance_type='diag',n_init = 3)
        gmm.fit(features)

        # dumping the trained gaussian model
        picklefile = path.split("-")[0]+".gmm"
        cPickle.dump(gmm,open(dest + picklefile,'wb'))
        print('+ modeling completed for speaker:',picklefile," with data point = ",features.shape)
        features = np.asarray(())

count = count + 1
"""
