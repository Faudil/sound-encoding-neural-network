from src.keywords.KerasClassifier import KerasClassifier

import numpy as np
from keras.layers import Dense, Conv2D, Activation, Flatten, BatchNormalization
from keras.models import Sequential
from librosa.feature import mfcc


class DigitClassifier(KerasClassifier):
    def __init__(self, file_path=None):
        super().__init__(file_path)

    def predict(self, x):
        # x = np.expand_dims(np.array([x]), axis=2)
        return self._model.predict(x)

    def build(self):
        model = Sequential()
        model.add(BatchNormalization())
        model.add(Conv2D(20, (2, 2), padding='same',
                         input_shape=(20, 32)))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(11, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self._model = model

    def transform(self, x, samplerate):
        to_process = mfcc(x, samplerate, n_mfcc=20)
        # to_process = pad_sequences(to_process, maxlen=32, padding='post')
        to_process = np.expand_dims(to_process, axis=2)
        return to_process
