from src.classifiers.KerasClassifier import KerasClassifier
from keras_preprocessing.sequence import pad_sequences
from keras.layers import Dense
from keras.models import Sequential
import numpy as np
import keras

from src.sound.SoundTransformer import SoundTransformer


class SpeechClassifier(KerasClassifier):
    def __init__(self, file_path=None):
        super().__init__(file_path)

    def build(self):
        self._model = Sequential()
        self._model.add(Dense(50, input_dim=216, activation='relu'))
        self._model.add(Dense(10, activation='relu'))
        self._model.add(Dense(2, activation='softmax'))
        opt = keras.optimizers.rmsprop(lr=0.00001, decay=1e-6)
        self._model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    def transform(self, x, samplerate):
        to_process = SoundTransformer.mfcc(x, samplerate)
        to_process = pad_sequences([to_process], maxlen=216, padding='post')[0]
        return np.array(to_process)

    def predict(self, x):
        x = np.array([x])
        return self._model.predict(x)
