import librosa as lbs
import soundfile as sf
import numpy as np
import os
from sklearn.mixture import GaussianMixture
import pickle

source = r"D:\ashish_gopal\sitw_database_related\database\train_sitw"
model_path = r"C:\Users\student\PycharmProjects\sitw_related\trained_models"

def_rate = 16000

files = [os.path.join(source, fname) for fname in os.listdir(source) if fname.endswith('.flac')]

for f in sorted(files):
    data, sr = sf.read(file=f)
    mfcc = lbs.feature.mfcc(y=data, sr=def_rate, n_mfcc=20)
    print(mfcc.shape)
    gmm = GaussianMixture(n_components=16, covariance_type='diag')
    gmm.fit(mfcc)
    path, speaker_name = str.rsplit(f, os.sep, 1)
    print(speaker_name[:-5])
    model_name = os.path.join(model_path, speaker_name[:-5]) + ".gmm"
    print(model_name)
    pickle.dump(gmm, open(model_name, 'wb'))
