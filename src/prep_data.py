from datetime import datetime, timedelta
import pandas as pd
from IPython import embed


def split_train_valid_test(args, df, df_inv, inv_dummy_diff,
                           test_start_datetime, eval_span):
    test_end_datetime = test_start_datetime + \
                        timedelta(days=eval_span)
    valid_start_datetime = test_start_datetime - \
                        timedelta(days=eval_span)
    valid_end_datetime = test_start_datetime
    train_end_datetime = valid_start_datetime

    first_day = df.index[0]
    train_start_datetime = datetime(first_day.year, first_day.month, first_day.day)

    df_train_start = df.index.get_loc(train_start_datetime)
    df_train_end = df.index.get_loc(train_end_datetime)
    df_train = df.iloc[df_train_start:df_train_end, :]

    df_valid_start = df.index.get_loc(valid_start_datetime)
    df_valid_end = df.index.get_loc(valid_end_datetime)
    df_valid = df.iloc[df_valid_start:df_valid_end, :]

    df_test_start = df.index.get_loc(test_start_datetime)
    df_test_end = df.index.get_loc(test_end_datetime)
    df_test = df.iloc[df_test_start:df_test_end, :]

    df_train = decimate_nonzero_train_data(df_train, df_test)
    df_train = decimate_past_train_data(args, df_train)

    # inverse
    train_inv_start_datetime = train_end_datetime - \
                               timedelta(days=inv_dummy_diff)
    train_inv_end_datetime = train_start_datetime - \
                             timedelta(days=inv_dummy_diff)
    df_inv_start = df_inv.index.get_loc(train_inv_start_datetime)
    df_inv_end = df_inv.index.get_loc(train_inv_end_datetime)
    df_inv_train = df_inv.iloc[df_inv_start:df_inv_end, :]
    df_inv_train = decimate_nonzero_train_data(df_inv_train, df_test)
    df_inv_train = decimate_past_train_data(args, df_inv_train)
    df_train = pd.concat([df_inv_train, df_train])

    df_train = df_train.sample(frac=1, random_state=42)
    X_train = df_train.iloc[:, :-1]
    y_train = df_train.iloc[:, -1]

    X_valid = df_valid.iloc[:, :-1]
    y_valid = df_valid.iloc[:, -1]

    X_test = df_test.iloc[:, :-1]
    y_test = df_test.iloc[:, -1]

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def decimate_nonzero_train_data(df_train, df_test):
    df_test_zero = df_test[df_test['adf flag'] == 0]
    df_test_one = df_test[df_test['adf flag'] == 1]
    df_test_two = df_test[df_test['adf flag'] == 2]
    test_frac_one_by_zero = df_test_one.shape[0] / df_test_zero.shape[0]
    test_frac_two_by_zero = df_test_two.shape[0] / df_test_zero.shape[0]

    df_train_zero = df_train[df_train['adf flag'] == 0]
    df_train_one = df_train[df_train['adf flag'] == 1]
    df_train_two = df_train[df_train['adf flag'] == 2]
    train_frac_one_by_zero = df_train_one.shape[0] / df_train_zero.shape[0]
    train_frac_two_by_zero = df_train_two.shape[0] / df_train_zero.shape[0]

    if train_frac_one_by_zero > test_frac_one_by_zero:
        print('======== decimate one')
        print('test: {}'.format(test_frac_one_by_zero))
        print('train: {}'.format(train_frac_one_by_zero))
        frac = test_frac_one_by_zero / train_frac_one_by_zero
        df_train_one = df_train_one.sample(frac=frac, random_state=42)

    if train_frac_two_by_zero > test_frac_two_by_zero:
        print('======== decimate two')
        print('test: {}'.format(test_frac_two_by_zero))
        print('train: {}'.format(train_frac_two_by_zero))
        frac = test_frac_two_by_zero / train_frac_two_by_zero
        df_train_two = df_train_two.sample(frac=frac, random_state=42)

    return pd.concat([df_train_zero, df_train_one, df_train_two])


def decimate_past_train_data(args, df_train):
    print('======== decimate past_data')
    df_train.sort_index(ascending=True, inplace=True)
    rows = df_train.shape[0]
    divs = 20
    div_number = [i//(rows//divs) for i in range(rows)]
    df_train['div_number'] =div_number
    dfs = []
    for i in range(divs):
        df = df_train[df_train['div_number'] == i]
        if i < int(divs*0.8):
            df = df.sample(frac=(i+1)/divs, random_state=42)
        dfs.append(df)
    df_train_mini = pd.concat(dfs)
    df_train_mini.drop(['div_number'], axis=1, inplace=True)
    return df_train_mini
