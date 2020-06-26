import numpy as np
import sounddevice as sd
import soundfile as sf
from src.sound.SoundRecorder import SoundRecorder


def main():
    sound_r = SoundRecorder(default_duration=2.0, samplerate=16000)
    print("recording")
    a = sound_r.record()
    print("playing")
    sd.play(a, blocking=True)
    print("done playing")
    print("Begin to register")
    sf.write("lalal.wav", a, samplerate=16000)
    print("Done writing")


if __name__ == '__main__':
    main()
