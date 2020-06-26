#!/usr/bin/env python3
import os


def parse_csv():
    with open("SPEAKERS.TXT") as f:
        for line in f.readlines()[1:]:
            line = line.strip()
            array = line.split("\t")
            if len(array) == 8:
                yield int(array[1].strip()), array[6].strip()


def move_files(id: int, gender: str):
    try:
        for f in os.listdir("clips/"):
            if f.endswith(".mp3"):
                # TODO: finich this method
                os.system(f"ffmpeg -i clips/{id}/{item}/{f} {'male' if gender == 'male' else 'female'}/{f.replace('.mp3', '.wav')}")
        print(id, gender)
    except Exception:
        pass


def main():
    """
    This file has issues, but because I don't need the data right now, I will fix it later
    """
    os.mkdir("./male")
    os.mkdir("./female")
    for id, gender in parse_csv():
        move_files(id, gender)


if __name__ == '__main__':
    main()
