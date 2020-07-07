import numpy as np


class VoiceModule:
    def __init__(self, module_name: str, labels: list, model):
        self._labels = labels
        self._name = module_name
        l = len(self._labels)
        self._lvectors = [np.array([1 if i == j else 0 for i in range(0, l)]) for j in range(0, l)]
        self._model = model

    def predict(self, sound, samplerate):
        r = self._model.predict(self._model.transform(sound, samplerate))
        label_name, loss = self.get_label_from_vector(r)
        return r, label_name, loss

    def get_label_from_vector(self, vector):
        """
        :param vector: The vector result you want to compute
        :return: The label name of the result and the distance between it and the perfect one_hot result
        """
        idx = int(np.argmax(vector))
        return self._labels[idx], 1 - np.linalg.norm(self._lvectors[idx] - vector)

    def label(self, idx: int) -> str:
        return self._labels[idx]

    def label_vector(self, label_name: str) -> np.array:
        return self._lvectors[self._labels.index(label_name)]

    def convert_prediction_to_json(self, r, label_name, _):
        res_json = {self._name: label_name}
        for (label, value) in zip(self._labels, r):
            res_json[label] = value.item()
        return res_json

    @property
    def model(self):
        return self._model
