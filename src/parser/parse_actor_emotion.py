#!/usr/bin/env python3
import os, shutil

emotion_index = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}


def move_files(id):
    for item in os.listdir(f"Actor_{id}"):
        if item.endswith(".wav"):
            t, a, emotion, *r = item.split('-')
            shutil.copy(f"Actor_{id}/{item}", emotion_index[emotion])
            print(id, emotion_index[emotion])


def main():
    for _, emotion_name in emotion_index.items():
        os.mkdir(emotion_name)
    for id, _ in enumerate(filter(lambda x: x.startswith("Actor_"), os.listdir(""))):
        move_files("{:02}".format(id + 1))


if __name__ == '__main__':
    main()
