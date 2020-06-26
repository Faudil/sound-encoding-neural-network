#!/usr/bin/env python3
import os
import librosa

from src.keywords.VoiceModule import VoiceModule
from src.keywords.digit_classifier.cnn1D import DigitClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np

labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]


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


def preprocess_all(nb_break):
    model = DigitClassifier()
    vm = VoiceModule('digit', ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], model)
    data = {
        "0": load_wav("zip/zero", nb_break),
        "1": load_wav("zip/one", nb_break),
        "2": load_wav("zip/two", nb_break),
        "3": load_wav("zip/three", nb_break),
        "4": load_wav("zip/four", nb_break),
        "5": load_wav("zip/five", nb_break),
        "6": load_wav("zip/six", nb_break),
        "7": load_wav("zip/seven", nb_break),
        "8": load_wav("zip/eight", nb_break),
        "9": load_wav("zip/nine", nb_break)
    }
    X, Y = preprare_wav(data, vm)
    np.save("x_digit.npy", X)
    np.save("y_digit.npy", Y)


def main():
    nb_break = 100
    model = DigitClassifier()
    vm = VoiceModule('digit', labels, model)
    print("Loading data")
    x_file = f"x_{'_'.join(labels)}"
    y_file = f"y_{'_'.join(labels)}"
    X, Y = load_data(x_file, y_file)
    X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.20, random_state=42)
    print("Loading done")
    while True:
        cmd = input("Enter epoch:")
        if cmd == 'stop':
            break
        elif cmd == 'reload':
            print("Loading data")
            X, Y = load_data(x_file, y_file)
            print("Loading done")
        elif cmd == "evaluate":
            print(model._model.evaluate(X_test, Y_test))
            y_pred = model._model.predict(X_test)
            matrix = confusion_matrix(Y_test.argmax(axis=1), y_pred.argmax(axis=1))
            print(matrix)
        elif cmd == "eval_all":
            print(model._model.evaluate(X, Y))
        else:
            vm.model.train(X_train, Y_train, batch_size=64, epoch=int(cmd), validation_data=(X_test, Y_test))
            vm.model.save("digit_test.model")
    print(model._model.evaluate(X_test, Y_test))


def test():
    model = DigitClassifier("digit_test.model")
    vm = VoiceModule('digit', labels, model)
    import sounddevice as sd
    X = load_wav("data/eight")
    for x, samplerate in X:
        sd.play(x, samplerate, blocking=True)
        print(vm.predict(x, samplerate))
        input("Press enter to play next sample")


if __name__ == '__main__':
    #preprocess_all(100000)
    # main()
    test()

