""" author = ____shikharparikh_____ """ 

import librosa as lbs
import numpy as np
import os
import soundfile as sd
import h5py

flac_file_path = r"D:\sitw_database_related\database\sitw_database.v4\dev\audio"
model_path = r"C:\Users\student\PycharmProjects\sitw_related\trained_models"

train_data_path = r"D:\sitw_database_related\database\train_sitw"
test_data_path = r"D:\sitw_database_related\database\test_sitw"

def_rate = 16000

files = [os.path.join(flac_file_path, fname) for fname in os.listdir(flac_file_path) if fname.endswith('.flac')]
large_files_count = 0

for f in sorted(files):
    # data, sr = lbs.core.load(path=f, sr=def_rate, mono=True, dtype=np.float64)
    data, sr = sd.read(file=f)
    duration = lbs.core.get_duration(data, sr=16000)
    # print(f, "Duration:", duration)
    # print(f, len(data))
    if duration >= 1000.0:
        large_files_count = large_files_count + 1
        # # print(f, duration)
        path, speaker_name = str.rsplit(f, os.sep, 1)
        # # print(len(data), len(data)*75/100, len(data)*25/100)
        train_data = data[0:int((len(data)*75/100))]
        test_data = data[int((len(data)*75/100)+1):len(data)]
        # print(len(train_data), len(test_data))
        train_save_path = os.path.join(train_data_path, speaker_name[:-5]) + ".flac"
        test_save_path = os.path.join(test_data_path, speaker_name[:-5]) + ".flac"
        print(train_save_path, test_save_path)
        sd.write(file=train_save_path, data=train_data, samplerate=def_rate)
        sd.write(file=test_save_path, data=test_data, samplerate=def_rate)
        # lbs.output.write_wav(path=train_save_path, y=train_data, sr=def_rate)
        # lbs.output.write_wav(path=test_save_path, y=train_data, sr=def_rate)

print(large_files_count)
