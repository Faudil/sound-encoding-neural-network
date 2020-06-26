class IClassifier:
    """
        This is an interface to show the method all models have to implement
    """
    def __init__(self, file_path=None):
        pass

    def train(self, X, Y, batch_size, epoch):
        pass

    def predict(self, x):
        pass

    def save(self):
        pass
