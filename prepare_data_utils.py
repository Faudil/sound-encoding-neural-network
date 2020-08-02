import os
import librosa
import numpy as np

from src.VoiceModule import VoiceModule


def load_wav(folder_path, nb_break=None):
    files = os.listdir(folder_path)
    i = 0
    has_limit = nb_break is not None
    for file in files:
        if has_limit and i > nb_break:
            break
        X, sample_rate = librosa.load("{}/{}".format(folder_path, file), res_type='kaiser_fast', sr=16000, offset=0)
        if len(X) > 1:
            i += 1
            yield X, sample_rate


def split_sample(buffer, samplerate, duration, step):
    offset = 0
    len_buffer = len(buffer)
    frame_duration = int(duration * samplerate)
    while offset + frame_duration <= len_buffer:
        sample = buffer[offset:offset + frame_duration]
        yield sample
        offset += int(step * samplerate)


def preprare_wav(data: dict, vm: VoiceModule, sample_duration, step):
    X, Y = [], []
    for key, values in data.items():
        key_vector = vm.label_vector(key)
        print("Doing", key)
        for value, sr in values:
            for sample in split_sample(value, sr, sample_duration, step):
                X.append(vm.model.transform(sample, sr))
                Y.append(key_vector)
    return np.array(X), np.array(Y)
