#!/usr/bin/env python3
import os, shutil


def parse_csv():
    with open("SPEAKERS.TXT") as f:
        for line in f.readlines():
            line = line.strip()
            if line.startswith(";"):
                continue
            array = line.split("|")
            yield int(array[0].strip()), array[1].strip()


def move_files(id: int, gender: str):
    try:
        for item in os.listdir(f"train-clean-100/{id}"):
            for f in os.listdir(f"train-clean-100/{id}/{item}"):
                if f.endswith(".flac"):
                    os.system(f"ffmpeg -i train-clean-100/{id}/{item}/{f} {'male' if gender == 'M' else 'female'}/{f.replace('.flac', '.wav')}")
        print(id, gender)
    except Exception:
        pass


def main():
    os.mkdir("./male")
    os.mkdir("./female")
    for id, gender in parse_csv():
        move_files(id, gender)


if __name__ == '__main__':
    main()
