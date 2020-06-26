import os
import librosa

from src.keywords.VoiceModule import VoiceModule
import numpy as np


def load_wav(folder_path, nb_break=None, samplerate=16000):
    files = os.listdir(folder_path)
    i = 0
    has_limit = nb_break is not None
    for file in files:
        if has_limit and i > nb_break:
            break
        X, sample_rate = librosa.load("{}/{}".format(folder_path, file), res_type='kaiser_fast', duration=1, sr=samplerate, offset=0)
        if len(X) > 1:
            i += 1
            yield X, sample_rate


def load_data(x_name, y_name):
    return np.load(x_name), np.load(y_name)


def preprare_wav(data: dict, vm: VoiceModule):
    X, Y = [], []
    for key, values in data.items():
        key_vector = vm.label_vector(key)
        print("Loading", key)
        for value, sr in values:
            X.append(vm.model.transform(value, sr))
            Y.append(key_vector)
    return np.array(X), np.array(Y)

