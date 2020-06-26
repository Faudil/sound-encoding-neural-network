import librosa
import sounddevice as sd
from scipy.io import wavfile
from src.sound.SoundTransformer import SoundTransformer


def main():
    noises = sd.rec(int(16000 * 0.5), 16000, blocking=True, channels=1)
    noises.squeeze()
    buffer, sr = librosa.load("dounia.wav", res_type='kaiser_fast', duration=3, sr=16000, offset=0)
    buffer = SoundTransformer.remove_noises(buffer, sr, noises)
    wavfile.write("clean.wav", 16000, buffer)
    #sd.play(buffer, 16000, blocking=True)



if __name__ == '__main__':
    main()
