import os
import pickle as cPickle
import numpy as np
from scipy.io.wavfile import read
from python_speech_features import mfcc
import warnings
warnings.filterwarnings("ignore")
import time


#path to training data
#source   = "/home/deeplearning/Downloads/development_set "
source = r"./testdata"

modelpath = "./trainedmodel"

test_file = "./datatest.txt"

#file_paths = open(test_file,'r')
test_files = [os.path.join(source,fname) for fname in os.listdir(source) if fname.endswith('.wav')]

gmm_files = [os.path.join(modelpath,fname) for fname in os.listdir(modelpath) if fname.endswith('.gmm')]

#Load the Gaussian Models
models    = [cPickle.load(open(fname,'rb')) for fname in gmm_files]
speakers   = [fname.split("/")[-1].split(".gmm")[0] for fname
              in gmm_files]
print(models)
# Read the test directory and get the list of test audio files
for f in test_files:
    #path = path.strip()
    #print(path)
    sr,audio = read(f)
        # extract 20 dimensional MFCC features
    vector = mfcc(audio,sr,0.025,0.01,20,appendEnergy=True)
    log_likelihood = np.zeros(len(models))
    for i in range(len(models)):
        gmm    = models[i]         #checking with each model one by one
        scores = np.array(gmm.score(vector))
        log_likelihood[i] = scores.sum()
    winner = np.argmax(log_likelihood)
    print(f,"\tdetected as - ", speakers[winner])
time.sleep(1.0)
