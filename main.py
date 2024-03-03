import fire
from LiidClassifierModel import LiidClassifierModel


class CLI(object):

    def __init__(self):
        self._model = LiidClassifierModel()

    def train(self, dataset):
        return self._model.train(dataset)

    def predict(self, dataset):
        return self._model.predict(dataset)


if __name__ == '__main__':
    fire.Fire(CLI)
