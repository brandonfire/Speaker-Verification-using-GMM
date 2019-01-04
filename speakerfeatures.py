import pandas as pd
import numpy as np
from sklearn import preprocessing
import python_speech_features as mfcc
from python_speech_features import delta


def extract_features(audio,rate):
    """extract 20 dimensional mfcc features from an audio, performs CMS and combines
    delta to make it 40 dimensional feature vector"""

    mfcc_feat = mfcc.mfcc(audio,rate, 0.025, 0.01,20,appendEnergy = True)

    mfcc_feat = preprocessing.scale(mfcc_feat)
    delta_mfcc_feat = delta(mfcc_feat,20)
    combined = np.hstack((mfcc_feat,delta_mfcc_feat))
    return combined
#
if __name__ == "__main__":
    print("In main, Call extract_features(audio,signal_rate) as parameters")
