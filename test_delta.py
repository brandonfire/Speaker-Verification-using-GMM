from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav

(rate,sig) = wav.read("/home/deeplearning/Downloads/development_set /belmontguy-20110426-geu/wav/b0152.wav")
mfcc_feat = mfcc(sig,rate)
d_mfcc_feat = delta(mfcc_feat, 20)
fbank_feat = logfbank(sig,rate)

print(fbank_feat)
