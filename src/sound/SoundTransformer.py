import librosa
import numpy as np


class SoundTransformer:
    def __init__(self):
        pass

    @staticmethod
    def mfcc(buffer, samplerate, dim=13):
        a = librosa.feature.mfcc(y=buffer, sr=samplerate, n_mfcc=dim)
        return np.mean(a, axis=0)

    @staticmethod
    def add_noises(buffer, samplerate, amplitude, noises=None):
        if not noises:
            noises = np.random.rand(len(buffer))
        else:
            l = len(buffer) // len(noises) + 1
            # Here we give noises the same size than buffer
            noises = np.tile(noises, l)[:len(buffer)]
        return buffer + (noises * amplitude)

    @staticmethod
    def remove_noises(buffer, samplerate, noises):
        l = len(buffer) // len(noises) + 1
        print(l)
        noises = noises.squeeze()
        noises = np.tile(noises, l)
        return buffer - noises[:len(buffer)]
