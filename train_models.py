import pickle as cPickle
import numpy as np
from scipy.io.wavfile import read
from sklearn.mixture import GaussianMixture
from sklearn import preprocessing
from speakerfeatures import extract_features
from python_speech_features import delta
from python_speech_features import mfcc
import warnings
warnings.filterwarnings("ignore")


#path to training data
source   = "/home/deeplearning/Downloads/development_set "

#path where training speakers will be saved
dest = "/home/deeplearning/Downloads/speaker_models"

train_file = "/home/deeplearning/Downloads/development_set_enroll.txt"


file_paths = open(train_file,'r')

count = 1

# Extracting features for each speaker (5 files per speakers)
features = np.asarray(())
for path in file_paths:
    path = path.strip()
    print(path)

    # read the audio
sr,audio = read(source + '/' + path)

    # extract 40 dimensional MFCC & delta MFCC features
    vector   = mfcc(audio,sr,0.025,0.01,20,appendEnergy=True)
    vector = preprocessing.scale(vector)
    if features.size == 0:
        features = vector
    else:
        features = np.vstack((features, vector))
    # when features of 5 files of speaker are concatenated, then do model training
    if count == 5:
        gmm = GaussianMixture(n_components = 16,covariance_type='diag',n_init = 3)
        gmm.fit(features)

        # dumping the trained gaussian model
        picklefile = path.split("-")[0]+".gmm"
        cPickle.dump(gmm,open(dest + picklefile,'wb'))
        print('+ modeling completed for speaker:',picklefile," with data point = ",features.shape)
        features = np.asarray(())
        count = 0
count = count + 1
