#!/usr/bin/env python3

import numpy as np

from src.classifiers.digit.cnn import DigitClassifier
from src.classifiers.speech.mlp import SpeechClassifier
from src.classifiers.word.cnn import WordClassifier
from src.sound.SoundRecorder import SoundRecorder
from src.VoiceModule import VoiceModule
from src.classifiers.gender.cnn_44kHz import GenderClassifier
import soundfile as sf
import sounddevice as sd


def test(modules: list):
    sr = SoundRecorder(default_duration=1.5)
    print("Recording")
    r = sr.record()
    print("Done")
    sd.play(r)
    a = np.squeeze(r)
    for module in modules:
        label, loss = module.predict(a, 22100)
        print(label, loss)
    return r


def main():
    model = GenderClassifier("sex.model")
    module = VoiceModule("gender", ["female", "male"], model)
    duration = 1  # seconds
    sr = SoundRecorder(default_duration=duration)
    print("Started recording")
    for i in range(25, 60):
        a = test([module, VoiceModule("yes_no", ["yes", "no"], WordClassifier("word.model"))])
        #sf.write("data/Faudil/sample_{}.wav".format(i), a, samplerate=44100)
        #sf.write("data/void/sample_{}.wav".format(i), a, samplerate=44100)
        #print(i)
    print("Done recording")
    print("Playing back")
    print("Done playing")


if __name__ == '__main__':
    main()

