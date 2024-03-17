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

    def log(self):
        # if not os.path.isdir("log"):
        #     os.mkdir("log")
        # log_file_name = f"log/myapp_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
        # logging.basicConfig(filename=log_file_name, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        logging.info("Start")
        self._model.log()
        return 'log'


if __name__ == '__main__':
    if not os.path.isdir("log"):    
        os.mkdir("log")
    log_file_name = f"log/myapp_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    logging.basicConfig(filename=log_file_name, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    fire.Fire(CLI)
