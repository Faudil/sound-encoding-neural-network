import numpy as np
import keras
from keras_preprocessing.sequence import pad_sequences
from src.FeatureExtractor import FeatureExtractor
from src.classifiers.KerasClassifier import KerasClassifier


class EmotionClassifier(KerasClassifier):
    def __init__(self, extractor_path, file_path=None):
        super().__init__(file_path)
        self._fe = FeatureExtractor(extractor_path)

    def predict(self, x):
        r = np.mean(self._model.predict(x), axis=0)
        return r

    def build(self):
        model = keras.Sequential()
        model.add(keras.layers.Dense(60, activation='relu'))
        model.add(keras.layers.Dense(units=8, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])
        self._model = model

    def transform(self, x, samplerate):
        x = self._fe.extract_from_wave_form(x, samplerate)
        x = x.flatten()
        x = pad_sequences([x], maxlen=256, padding='post')[0]
        return np.array([x])
