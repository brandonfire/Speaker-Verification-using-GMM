# Testing source code

import os
import _pickle as cPickle
import numpy as np
import soundfile as sf
import librosa as lbs
from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")

def_rate = 16000


def get_MFCC(sr, audio):
    features = lbs.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    feat = np.asarray(())
    for i in range(features.shape[0]):
        temp = features[i, :]
        if np.isnan(np.min(temp)):
            continue
        else:
            if feat.size == 0:
                feat = temp
            else:
                feat = np.vstack((feat, temp))
    features = feat
    features = preprocessing.scale(features)
    return features


# path to testing data
sourcepath = r"D:\ashish_gopal\sitw_database_related\database\test_sitw"

# path to saved models
modelpath = r"C:\Users\student\PycharmProjects\sitw_related\trained_models"

gmm_files = [os.path.join(modelpath, fname) for fname in os.listdir(modelpath) if fname.endswith('.gmm')]
print(len(gmm_files))
models = [cPickle.load(open(fname, 'rb')) for fname in gmm_files]
speakers = [fname.split("\\")[-1].split(".gmm")[0] for fname in gmm_files]
files = [os.path.join(sourcepath, f) for f in os.listdir(sourcepath) if f.endswith(".flac")]
print(len(files))

for f in files:
    print(f.split("\\")[-1])
    audio, sr = sf.read(file=f)
    features = get_MFCC(sr, audio)
    print(features.shape)
    scores = None
    log_likelihood = np.zeros(len(models))
    for i in range(len(models)):
        gmm = models[i]
        scores = np.array(gmm.score(features))
        log_likelihood[i] = scores.sum()
    winner = np.argmax(log_likelihood)
    print(f, speakers[winner])

# print("\tDetected as - ", speakers[winner], "\n\tscores:female ", log_likelihood[0], ",male ", log_likelihood[1],"\n")
