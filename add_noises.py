import librosa
import sounddevice as sd
from src.sound.SoundTransformer import SoundTransformer


def main():
    buffer, sr = librosa.load("dounia.wav", res_type='kaiser_fast', duration=3, sr=16000, offset=0)
    buffer = SoundTransformer.add_noises(buffer, sr, 0.01)
    sd.play(buffer, 16000, blocking=True)


if __name__ == '__main__':
    main()
