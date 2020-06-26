import src.tensorflow.vggish_keras as vggish_keras
import src.tensorflow.vggish_input as vggish_input
import numpy as np


class FeatureExtractor:
    def __init__(self, checkpoint_path):
        self.model = vggish_keras.get_vggish_keras()
        self.model.load_weights(checkpoint_path)

    def extract_from_wav(self, file_path):
        waveform = vggish_input.wavfile_to_examples(file_path)

        return self._extract(waveform)

    def extract_from_wave_form(self, wave_form, samplerate):
        x = vggish_input.waveform_to_examples(wave_form, samplerate)
        return self._extract(x)

    def _extract(self, x):
        x = np.expand_dims(x, axis=3)
        return self.model.predict(x)
