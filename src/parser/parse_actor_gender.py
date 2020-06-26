#!/usr/bin/env python3
import os, shutil


def move_files(id, gender: str):
    for item in os.listdir(f"Actor_{id}"):
        if item.endswith(".wav"):
            shutil.copy(f"Actor_{id}/{item}", gender)
    print(id, gender)


def main():
    os.mkdir("./male")
    os.mkdir("./female")
    for id, _ in enumerate(filter(lambda x: x.startswith("Actor_"), os.listdir(""))):
        move_files("{:02}".format(id + 1), "male" if (id + 1) % 2 != 0 else "female")


if __name__ == '__main__':
    main()
