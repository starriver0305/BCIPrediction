import numpy as np
import pandas as pd
import configparser
import sys
from sklearn.model_selection import StratifiedKFold
from models import model_dict
from utils import evaluation

config = configparser.ConfigParser()
config.read('../config.ini')
data_path = config['path']['data_path']
loc_label = ['EEG.AF3', 'EEG.F7', 'EEG.F3', 'EEG.FC5', 'EEG.T7', 'EEG.P7', 'EEG.O1', 'EEG.O2', 'EEG.P8', 'EEG.T8',
             'EEG.FC6', 'EEG.F4', 'EEG.F8', 'EEG.AF4']

args = sys.argv
model_name = args[1]


def read():
    df = pd.read_csv(data_path)
    X, Y = [], []
    X = df.loc[:, loc_label].values
    tag = 0
    timestamp = -1
    for i in range(len(X)):
        cur_time = df.loc[i, 'Timestamp']
        if timestamp == -1:
            timestamp = cur_time
        elif cur_time - timestamp >= 8 and tag <= 1:
            tag += 1
            timestamp = cur_time
        Y.append(int(tag))
    per = np.random.permutation(X.shape[0])  # 打乱后的行号
    new_X = np.array(X)[per]
    new_Y = np.array(Y)[per]
    return np.array(new_X), np.array(new_Y)


if __name__ == '__main__':
    x_data, y_data = read()
    n_splits = 10
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_index, test_index in kf.split(x_data, y_data):
        X_train, Y_train = x_data[train_index], y_data[train_index]
        X_test, Y_test = x_data[test_index], y_data[test_index]
        model = model_dict[model_name]
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        evaluation.evaluate(Y_test, Y_pred, model_name)


