import keras
import numpy as np
from keras_preprocessing.sequence import pad_sequences
from keras.layers import Dense, Dropout, Conv1D, Activation, Flatten, MaxPooling1D, BatchNormalization, LSTM
from keras.models import Sequential
from python_speech_features import mfcc

from src.sound.SoundTransformer import SoundTransformer
from src.classifiers.KerasClassifier import KerasClassifier


class DigitClassifier(KerasClassifier):
    def __init__(self, file_path=None):
        super().__init__(file_path)

    def predict(self, x):
        #x = np.expand_dims(np.array([x]), axis=2)
        return self._model.predict(x)

    def build(self):
        model = Sequential()
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dense(10, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self._model = model

    def transform(self, x, samplerate):
        to_process = mfcc(x, samplerate, nfft=551)
        to_process = np.mean(to_process, axis=1)
        to_process = pad_sequences([to_process], maxlen=128, padding='post')[0]
        # to_process = np.expand_dims(to_process, axis=2)
        return to_process
