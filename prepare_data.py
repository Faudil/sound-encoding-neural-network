#!/usr/bin/env python3
import argparse


from src.VoiceModule import VoiceModule
from src.classifiers.gender.vggish_tuned import GenderClassifier

import numpy as np
from prepare_data_utils import preprare_wav, load_wav




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
