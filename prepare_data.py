#!/usr/bin/env python3
import argparse
import os
import librosa

from src.VoiceModule import VoiceModule
from src.classifiers.gender.vggish_tuned import GenderClassifier
from src.vad import contains_voice
import numpy as np


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
    while offset + frame_duration < len_buffer:
        sample = buffer[offset:offset + frame_duration]
        if contains_voice(sample * 32767, samplerate):
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


def preprocess_all(dir, files, nb_break, sample_duration, step):
    model = GenderClassifier("src/tensorflow/vggish_weights.ckpt")
    vm = VoiceModule('__process__', files, model)
    data = {f: load_wav(f"{dir}/{f}", nb_break) for f in files}
    X, Y = preprare_wav(data, vm, sample_duration, step)
    np.save(f"x_{'_'.join(files)}.npy", X)
    np.save(f"y_{'_'.join(files)}.npy", Y)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-dir", help="dir where files are stored", type=str)
    ap.add_argument("-duration", help="duration of sample", type=float)
    ap.add_argument("-step", help="size of step when splitting sample", type=float)
    ap.add_argument("-n", help="Number of files to process", type=int)
    ap.add_argument('files', help="files to parse", nargs=argparse.REMAINDER)
    return ap.parse_args()


def main():
    args = parse_args()
    for f in args.files:
        print(f)
    print(args.n)
    preprocess_all(args.dir, args.files, args.n, args.duration, args.step)


if __name__ == '__main__':
    main()
