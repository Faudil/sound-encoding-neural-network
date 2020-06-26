import librosa
import numpy as np
import matplotlib.pyplot as plt
from src.sound.SoundTransformer import SoundTransformer
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav


def main():
    sig, rate = librosa.load("chloe_hq1.wav", res_type='kaiser_fast', duration=2.5, sr=16000,
                          offset=0)
    mfcc_librosa = librosa.feature.mfcc(sig, rate)
    mfcc_feat = np.mean(mfcc(sig, rate), axis=1)
    # d_mfcc_feat = delta(mfcc_feat, 2)
    # fbank_feat = logfbank(sig, rate)
    plt.plot(mfcc_feat)
    plt.show()
    plt.plot(mfcc_librosa)
    plt.show()


if __name__ == '__main__':
    main()
