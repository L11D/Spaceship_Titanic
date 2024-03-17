import warnings

import numpy as np
import pandas as pd
import logging
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')
import os

logger = logging.getLogger(__name__)
out_of_range_value = -9999
target = 'Transported'

params = {
    'learning_rate': 0.05464365018363548,
    'depth': 5,
    'iterations': 322,
    'l2_leaf_reg': 0.937536434535921,
    'random_strength': 0.8110316739919927,
    'boost_from_average': False,
    'subsample': 0.6665211600601834
}


class LiidClassifierModel(object):
    """Model for predicting Spaceship Titanic. Based on CatBoostClassifier
        """

    def __imput_mis_val(self, data):
        """
        Imputes missing values
        :param data: pd.DataFrame
        :return: pd.DataFrame
        """
        logging.info("Imputing missing data")
        ###############################################

        data = data.drop(columns='Name')  # drop useless feature
        data = data.drop(columns='PassengerId')  # drop useless feature
        data = data.drop(columns='Destination')  # drop useless feature

        # imput with mode value
        data.RoomService = data.RoomService.fillna(0)
        data.FoodCourt = data.FoodCourt.fillna(0)
        data.ShoppingMall = data.ShoppingMall.fillna(0)
        data.Spa = data.Spa.fillna(0)
        data.VRDeck = data.VRDeck.fillna(0)
        data.VIP = data.VIP.fillna(data.VIP.mode()[0])
        data.HomePlanet = data.HomePlanet.fillna(data.HomePlanet.mode()[0])

        # imput with mode out_of_range_value
        data.Cabin = data.Cabin.fillna(f'{out_of_range_value}/{out_of_range_value}/{out_of_range_value}')
        data.Age = data.Age.fillna(out_of_range_value)

        # imput CryoSleep based on passenger Expenditure
        data.loc[(data['CryoSleep'].isnull()) & (
                data['RoomService'] + data['FoodCourt'] + data['Spa'] + data['ShoppingMall'] + data[
            'VRDeck']) > 0, 'CryoSleep'] = False
        data.loc[data['CryoSleep'].isnull(), 'CryoSleep'] = True
        data.CryoSleep = pd.to_numeric(data.CryoSleep)

        ###############################################
        logging.info("Imputing missing data completed")
        return data

    def __unpack_features(self, data):
        """
        Unpack features from dataframe
        :param data: pd.DataFrame
        :return: pd.DataFrame
        """
        logging.info("Unpacking features")
        ###############################################
        # unpack deck from Cabin
        data['deck'] = data.Cabin.map(lambda x: x.split('/')[0]).map(
            lambda x: x if x != str(out_of_range_value) else 'Z')

        # unpack side from Cabin
        data['side'] = data.Cabin.map(lambda x: x.split('/')[2]).map(
            lambda x: x if x != str(out_of_range_value) else 'Z')
        data = data.drop(columns='Cabin')

        ###############################################
        logging.info("Unpacking features completed")
        return data

    def __encoding_range_features(self, data):
        """
        Encodes range features
        :param data: pd.DataFrame
        :return: pd.DataFrame
        """
        logging.info("Replacing range features")
        ###############################################

        # encoding deck feature
        mapdict = {val: i for i, val in enumerate(sorted(set(data['deck'].values)))}
        data['deck'] = data['deck'].map(mapdict)

        ############################################### 
        logging.info("Replacing range features completed")
        return data

    def __one_hot(self, data):
        """
        Takes a dataframe and encodes it with one-hot
        :param data: pd.DataFrame
        :return: pd.DataFrame
        """
        logging.info("One hot HomePlanet, side features")
        ###############################################

        # one hot encoding HomePlanet, side features
        cat_features = ['HomePlanet', 'side']
        for cat_f in cat_features:
            col = data[cat_f]
            for val in col.dropna().unique():
                new_col_data = col.map(lambda x: np.nan if pd.isna(x) else x == val)
                data[cat_f + '_' + val] = new_col_data
            data = data.drop(columns=cat_f)

        ###############################################
        logging.info("One hot HomePlanet, side features complete")
        return data

    def __add_fetures(self, data):
        """
        Adds new features to the dataset
        :param data: pd.DataFrame
        :return: pd.DataFrame
        """
        # add Expenditure feature
        exp_features = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        data["Expenditure"] = data[exp_features].sum(axis="columns")
        return data

    def __prepare_data(self, data):
        """
        Prepare the data for training and testing
        :param data: pd.DataFrame
        :return: pd.DataFrame
        """
        logging.info("Preparing dataset")
        ###############################################

        data = self.__imput_mis_val(data)
        data = self.__unpack_features(data)
        data = self.__encoding_range_features(data)
        data = self.__one_hot(data)
        data = self.__add_fetures(data)

        ###############################################
        logging.info("Preparing dataset completed")
        return data

    def __fit_save(self, X, y, params):
        """
        Fit the model and save it
        :param X:
        :param y:
        :param params: params of CatBoostClassifier
        :return:
        """
        model = CatBoostClassifier(
            # **params,
            random_seed=42,
            logging_level='Silent')

        categorical_features_indices = np.where(X.dtypes == object)[0]
        model.fit(X, y, cat_features=categorical_features_indices)

        if not os.path.isdir("data"):
            os.mkdir("data")
        if not os.path.isdir("data/model"):
            os.mkdir("data/model")
        model.save_model('data/model/saved_model.cbm')
        logging.info("Model saved to data/model/saved_model.cbm")

    def train(self, dataset):
        """
        Train the model on the given dataset and save the trained model
        :param dataset: path to the dataset
        """
        logging.info("Train started")
        train_df = pd.DataFrame()
        try:
            logging.info("Reading dataset")
            train_df = pd.read_csv(dataset)
        except:
            logging.critical("Reading dataset failed")
            return 'Reading dataset failed'
        logging.info("Dataset has been read")
        y = train_df[target]
        train_df = train_df.drop(columns=target)
        train_df = self.__prepare_data(train_df)
        self.__fit_save(train_df, y, params)
        logging.info("Train completed")
        return 'train completed'

    def predict(self, dataset):
        """
        Predict the class of the given dataset and save prediction
        :param dataset: path to the dataset
        :return:
        """
        logging.info("Predict started")
        test_df = pd.DataFrame()
        try:
            logging.info("Reading dataset")
            test_df = pd.read_csv(dataset)
        except:
            logging.critical("Reading dataset failed")
            return 'Reading dataset failed'
        logging.info("Dataset has been read")

        passengerIds = test_df.PassengerId
        test_df = self.__prepare_data(test_df)

        model = CatBoostClassifier(
            # **params,
            random_seed=42,
            logging_level='Silent')

        try:
            logging.info("Loading model")
            model.load_model('data/model/saved_model.cbm', 'cbm')
        except:
            logging.info("Not found saved model")
            return 'not found saved model'
        logging.info("Model has been loaded")

        predictions = model.predict(test_df)

        if not os.path.isdir("data"):
            os.mkdir("data")

        out = pd.DataFrame(data={'PassengerId': passengerIds.values, 'Transported': predictions})
        out.to_csv('data/results.csv', index=False)
        logging.info("Predictions saved to results.csv")
        return 'predictions saved to results.csv'
