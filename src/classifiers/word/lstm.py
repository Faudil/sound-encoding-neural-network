import keras
import numpy as np
from keras_preprocessing.sequence import pad_sequences
from keras.layers import Dense, Dropout, Conv1D, Activation, Flatten, MaxPooling1D, BatchNormalization, LSTM
from keras.models import Sequential

from src.sound.SoundTransformer import SoundTransformer
from src.classifiers.KerasClassifier import KerasClassifier


class WordClassifier(KerasClassifier):
    def __init__(self, file_path=None):
        super().__init__(file_path)

    def predict(self, x):
        x = np.expand_dims(np.array([x]), axis=2)
        return self._model.predict(x)

    def build(self):
        model = Sequential()
        model.add(Conv1D(56, 3, padding='same'))
        model.add(Activation('relu'))
        model.add(Conv1D(56, 3, padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling1D(pool_size=4))
        model.add(BatchNormalization())
        model.add(LSTM(9))
        model.add(Dense(3))
        model.add(Activation('softmax'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self._model = model

    def transform(self, x, samplerate):
        to_process = SoundTransformer.mfcc(x, samplerate)
        to_process = pad_sequences([to_process], maxlen=56, padding='post')[0]
        return to_process
