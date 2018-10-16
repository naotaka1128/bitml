import csv
import datetime
import logging
import pandas as pd
import numpy as np
from itertools import chain
from hyperopt import fmin, tpe, hp
from src.generate_data import generate_data
from src.prep_data import split_train_valid_test
from src.train_model import exec_cross_validation
from src.data.bitmex import load_price_data, load_inv_price_data

ALGO_NAME = 'LightGBM'

TEST_START_DATETIMES = [
    datetime.datetime(2018,  8, 1),
    datetime.datetime(2018,  7, 1),
    datetime.datetime(2018,  6, 1),
]
EVAL_SPAN = 30
INV_DUMMY_DIFF = 10000

def init_params():
    if ALGO_NAME == 'LightGBM':
        return {
            'algo_name': hp.choice('algo_name', ['LightGBM']),
            'n_folds': hp.choice('n_folds', [3]),
            'delay_hour': hp.choice('delay_hour', [1]),
            'flag_threshold': hp.choice('flag_threshold', [0.01]),
            'threshold_outlier': hp.choice('threshold_outlier', [5]),
            'delete_correlate_columns': hp.choice('delete_correlate_columns', [0.99]),
        }
    elif ALGO_NAME == 'CatBoost':
        return {
            'algo_name': hp.choice('algo_name', ['CatBoost']),
            'n_folds': hp.choice('n_folds', [3]),
            'delay_hour': hp.choice('delay_hour', [1]),
            'flag_threshold': hp.choice('flag_threshold', [0.01]),
            'threshold_outlier': hp.choice('threshold_outlier', [5]),
            'delete_correlate_columns': hp.choice('delete_correlate_columns', [0.99]),
        }


def exec_test(args):
    delay_seconds = args['delay_hour'] * 60 * 60  # (sec)
    tick_seconds = 15 * 60  # (sec)
    delay = delay_seconds // tick_seconds

    df_raw = load_price_data()
    df, df_real_price = generate_data(args, df_raw, delay,
                                      15, args['flag_threshold'])

    df_inv_raw = load_inv_price_data(INV_DUMMY_DIFF)
    df_inv, _ = generate_data(args, df_inv_raw, delay,
                              15, args['flag_threshold'], df)

    scores = []
    test_loglosses = []

    for test_start_datetime in TEST_START_DATETIMES:
        X_train, X_valid, X_test, y_train, y_valid, y_test = \
            split_train_valid_test(args, df, df_inv,
                             INV_DUMMY_DIFF, test_start_datetime, EVAL_SPAN)

        score, test_logloss, model = exec_cross_validation(
            args, X_train, y_train, X_valid, y_valid,
            X_test, y_test)
        scores.append(score)

    return np.average(scores)


if __name__ == '__main__':
    params = init_params()
    best = fmin(exec_test, params, algo=tpe.suggest, max_evals=20)
