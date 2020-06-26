import tensorflow.keras
import numpy as np
from keras_preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Activation, Flatten, MaxPooling1D, BatchNormalization
from tensorflow.keras.models import Sequential

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

        model.add(Conv1D(56, 5, padding='same',
                         input_shape=(56, 1)))
        model.add(Activation('relu'))

        model.add(Conv1D(56, 5, padding='same'))
        model.add(Activation('relu'))
        model.add(Conv1D(56, 5, padding='same'))
        model.add(Activation('relu'))
        model.add(Conv1D(56, 5, padding='same'))
        model.add(Activation('relu'))

        model.add(Conv1D(56, 5, padding='same'))
        model.add(Activation('relu'))
        model.add(Conv1D(56, 5, padding='same'))
        model.add(Activation('relu'))
        model.add(Conv1D(56, 5, padding='same'))
        model.add(Activation('relu'))

        model.add(Flatten())
        model.add(Dense(2))
        model.add(Activation('softmax'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self._model = model

    def transform(self, x, samplerate):
        to_process = SoundTransformer.mfcc(x, samplerate)
        to_process = pad_sequences([to_process], maxlen=56, padding='post')[0]
        return to_process
