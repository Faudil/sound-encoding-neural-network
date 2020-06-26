from src.keywords.KerasClassifier import KerasClassifier

import numpy as np
from keras_preprocessing.sequence import pad_sequences
from keras.layers import Dense, Activation, Flatten, BatchNormalization
from keras.models import Sequential
from librosa.feature import mfcc


class DigitClassifier(KerasClassifier):
    def __init__(self, file_path=None):
        super().__init__(file_path)

    def predict(self, x):
        x = np.expand_dims(np.array([x]), axis=2)
        return self._model.predict(x)

    def build(self):
        model = Sequential()
        model.add(BatchNormalization())
        model.add(Dense(32))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(10, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self._model = model

    def transform(self, x, samplerate):
        to_process = mfcc(x, samplerate, n_mfcc=13)
        to_process = pad_sequences(to_process, maxlen=32, padding='post')
        # to_process = np.expand_dims(to_process, axis=2)
        return to_process
