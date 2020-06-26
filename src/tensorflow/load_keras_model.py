import src.tensorflow.vggish_keras as vggish_keras
import src.tensorflow.vggish_input as vggish_input
import src.tensorflow.vggish_postprocess as vggish_postprocess
import numpy as np
import keras
from random import shuffle
from src.FeatureExtractor import FeatureExtractor
import os


def load_folder(folder_path):
    for i, item in enumerate(os.listdir(folder_path)):
        if i >= 800:
            break
        yield vggish_input.wavfile_to_examples(f"{folder_path}/{item}")


def _get_examples_batch():
    """Returns a shuffled batch of examples of all audio classes.

    Note that this is just a toy function because this is a simple demo intended
    to illustrate how the training code might work.

    Returns:
        a tuple (features, labels) where features is a NumPy array of shape
        [batch_size, num_frames, num_bands] where the batch_size is variable and
        each row is a log mel spectrogram patch of shape [num_frames, num_bands]
        suitable for feeding VGGish, while labels is a NumPy array of shape
        [batch_size, num_classes] where each row is a multi-hot label vector that
        provides the labels for corresponding rows in features.
    """
    # Make a waveform for each class.
    num_seconds = 2.5
    sr = 16000    # Sampling rate.

    male_example = np.concatenate(tuple(load_folder("../../data/male")))
    male_labels = np.array([[1, 0]] * male_example.shape[0])
    female_example = np.concatenate(tuple(load_folder("../../data/female")))
    female_labels = np.array([[0, 1]] * female_example.shape[0])

    # Shuffle (example, label) pairs across all classes.
    all_examples = np.concatenate((male_example, female_example))
    all_labels = np.concatenate((male_labels, female_labels))
    labeled_examples = list(zip(all_examples, all_labels))
    shuffle(labeled_examples)
    # Separate and return the features and labels.
    features = [example for (example, _) in labeled_examples]
    labels = [label for (_, label) in labeled_examples]
    return features, labels


def create_model():
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(30, activation='relu'))
    model.add(keras.layers.Dense(units=2, activation='softmax'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=["accuracy"])
    return model


def train(model, fe):
    print("Loading wav files")
    feature, label = _get_examples_batch()
    #a = np.expand_dims(feature, axis=3)
    print("Extracting feature")
    processed_data = fe._extract(feature)
    #for _ in range(0, 30):
    while True:
        s = input("continue ?")
        if s == 'n':
            break
        model.fit(x=processed_data, y=np.array(label), epochs=30)
        model.save("transfer_learned.model")

def main():
    # input = vggish_input.wavfile_to_examples(f"../../data/male/03-01-01-01-01-01-01.wav")
    # print(a.shape)
    fe = FeatureExtractor('vggish_weights.ckpt')
    print("Feature extractor loaded")
    model = create_model()
    print("model created")
    train(model, fe)


if __name__ == '__main__':
    main()
