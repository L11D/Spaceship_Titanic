import fire
from LiidClassifierModel import LiidClassifierModel
import logging
import os
import datetime

class CLI(object):

    def __init__(self):
        self._model = LiidClassifierModel()

    def train(self, dataset):
        return self._model.train(dataset)

    def predict(self, dataset):
        return self._model.predict(dataset)


if __name__ == '__main__':
    if not os.path.isdir("data"):
            os.mkdir("data")
    if not os.path.isdir("data/log"):
        os.mkdir("data/log")
    log_file_name = f"data/log/log_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    logging.basicConfig(filename=log_file_name, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    fire.Fire(CLI)

# docker run -v $(pwd)/dataset:/SpaceShipTitanic/data spaceship_titanic_app train data/train.csv
