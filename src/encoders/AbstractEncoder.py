import tensorflow.keras as keras
import numpy as np
from keras_preprocessing.sequence import pad_sequences
from librosa.feature import mfcc

from src.classifiers.KerasClassifier import KerasClassifier


class AbstractEncoder:
    """
    Abstract class to avoid code duplication for keras based model
    """
    def __init__(self, file_path=None):
        self._model = None
        if not file_path:
            self.build()
        else:
            self.load(file_path)

    def build(self):
        raise Exception("This method is abstract and cannot be called")

    def load(self, file_path):
        self._model = keras.models.load_model(file_path)

    def train(self, X, Y, batch_size=32, epoch=720, validation_data=None):
        self._model.fit(X, Y, batch_size=batch_size, epochs=epoch, validation_data=validation_data)

    def predict(self, x):
        return self._model.predict(x)

    def transform(self, x, samplerate):
        raise Exception("This method is abstract and cannot be called")

    def save(self, file_name):
        self._model.save(file_name)
