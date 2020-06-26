import keras
import numpy as np
from keras_preprocessing.sequence import pad_sequences
from keras.layers import Dense, Dropout, Conv1D, Activation, Flatten, MaxPooling1D
from keras.models import Sequential

from src.sound.SoundTransformer import SoundTransformer
from src.classifiers.KerasClassifier import KerasClassifier


class GenderClassifier(KerasClassifier):
    def __init__(self, file_path=None):
        super().__init__(file_path)

    def predict(self, x):
        x = np.expand_dims(np.array([x]), axis=2)
        return self._model.predict(x)

    def build(self):
        model = Sequential()

        model.add(Conv1D(256, 5, padding='same',
                         input_shape=(216, 1)))
        model.add(Activation('relu'))
        model.add(Conv1D(128, 5, padding='same'))
        model.add(Activation('relu'))
        model.add(Dropout(0.1))
        model.add(MaxPooling1D(pool_size=8))
        model.add(Conv1D(128, 5, padding='same'))
        model.add(Activation('relu'))
        model.add(Conv1D(128, 5, padding='same'))
        model.add(Activation('relu'))
        model.add(Conv1D(128, 5, padding='same'))
        model.add(Activation('relu')),
        model.add(Conv1D(128, 5, padding='same'))
        model.add(Activation('relu'))
        model.add(Conv1D(128, 5, padding='same'))
        model.add(Activation('relu')),
        model.add(Dropout(0.2))
        model.add(Conv1D(128, 5, padding='same'))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(2))
        model.add(Activation('softmax'))
        opt = keras.optimizers.rmsprop(lr=0.00001, decay=1e-6)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        self._model = model

    def transform(self, x, samplerate):
        to_process = SoundTransformer.mfcc(x, samplerate)
        to_process = pad_sequences([to_process], maxlen=96, padding='post')[0]
        return to_process
