import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
import os

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

    def fix_mis_val(self, data):
        data = data.drop(columns='Name')  # drop useless feature

        data.RoomService = data.RoomService.fillna(0)
        data.FoodCourt = data.FoodCourt.fillna(0)
        data.ShoppingMall = data.ShoppingMall.fillna(0)
        data.Spa = data.Spa.fillna(0)
        data.VRDeck = data.VRDeck.fillna(0)

        data.VIP = data.VIP.fillna(data.VIP.mode()[0])  # test
        data.Destination = data.Destination.fillna(data.Destination.mode()[0])
        data.HomePlanet = data.HomePlanet.fillna(data.HomePlanet.mode()[0])
        data.Cabin = data.Cabin.fillna(f'{out_of_range_value}/{out_of_range_value}/{out_of_range_value}')
        data.Age = data.Age.fillna(out_of_range_value)

        data.loc[(data['CryoSleep'].isnull()) & (
                data['RoomService'] + data['FoodCourt'] + data['Spa'] + data['ShoppingMall'] + data[
            'VRDeck']) > 0, 'CryoSleep'] = False
        data.loc[data['CryoSleep'].isnull(), 'CryoSleep'] = True
        data.CryoSleep = pd.to_numeric(data.CryoSleep)
        return data

    def unpack_features(self, data):
        data = data.drop(columns='PassengerId')
        data['deck'] = data.Cabin.map(lambda x: x.split('/')[0]).map(
            lambda x: x if x != str(out_of_range_value) else 'Z')
        data['side'] = data.Cabin.map(lambda x: x.split('/')[2]).map(
            lambda x: x if x != str(out_of_range_value) else 'Z')
        data = data.drop(columns='Cabin')
        return data

    def replace_range_features(self, data):
        mapdict = {val: i for i, val in enumerate(sorted(set(data['deck'].values)))}
        data['deck'] = data['deck'].map(mapdict)
        return data

    def replace_range_features(self, data):
        mapdict = {val: i for i, val in enumerate(sorted(set(data['deck'].values)))}
        data['deck'] = data['deck'].map(mapdict)
        return data

    def one_hot(self, data):
        cat_features = ['HomePlanet', 'Destination', 'side']
        for cat_f in cat_features:
            col = data[cat_f]
            for val in col.dropna().unique():
                new_col_data = col.map(lambda x: np.nan if pd.isna(x) else x == val)
                data[cat_f + '_' + val] = new_col_data
            data = data.drop(columns=cat_f)
        return data

    def prepare_data(self, data):
        data = self.fix_mis_val(data)
        data = self.unpack_features(data)
        data = self.replace_range_features(data)
        data = self.one_hot(data)
        return data

    def fit_save(self, X, y, params):
        model = CatBoostClassifier(
            **params,
            random_seed=42,
            logging_level='Silent')

        categorical_features_indices = np.where(X.dtypes == object)[0]
        model.fit(X, y, cat_features=categorical_features_indices)

        if not os.path.isdir("data"):
            os.mkdir("data")
        if not os.path.isdir("data/model"):
            os.mkdir("data/model")
        model.save_model('data/model/saved_model.cbm')

    def train(self, dataset):
        train_df = pd.read_csv(dataset)
        y = train_df[target]
        train_df = train_df.drop(columns=target)
        train_df = self.prepare_data(train_df)
        self.fit_save(train_df, y, params)
        return 'train completed'

    def predict(self, dataset):
        test_df = pd.read_csv(dataset)
        PassengerIds = test_df.PassengerId
        test_df = self.prepare_data(test_df)

        model = CatBoostClassifier(
            **params,
            random_seed=42,
            logging_level='Silent')

        try:
            model.load_model('data/model/saved_model.cbm', 'cbm')
        except:
            return 'not found saved model'

        predictions = model.predict(test_df)

        if not os.path.isdir("data"):
            os.mkdir("data")

        out = pd.DataFrame(data={'PassengerId': PassengerIds.values, 'Transported': predictions})
        out.to_csv('data/results.csv', index=False)

        return 'predictions saved to /data/results.csv'
