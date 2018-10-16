import io
import datetime
import pandas as pd
from IPython import embed

def load_raw_data():
    ######## latest data from bitmex
    df = pd.read_csv('src/data/files/bitmex_15min.csv')
    df['datetime'] = pd.to_datetime(df["datetime"])
    df.set_index('datetime', inplace=True)

    ######## past data from CoinBase (price as JPY)
    df_past = pd.read_csv('src/data/files/coinbase_15min.csv')
    df_past['datetime'] = pd.to_datetime(df_past["datetime"])
    df_past.set_index('datetime', inplace=True)
    df_past = jpy_to_usd(df_past,
                                   ['open', 'high', 'low', 'close'])

    transfer_end_datetime = datetime.datetime(2017, 7, 1)
    transfer_end_idx = df_past.index.get_loc(transfer_end_datetime)
    df_past = df_past.iloc[:transfer_end_idx, :]
    target_start_idx = df.index.get_loc(transfer_end_datetime)
    df = df.iloc[target_start_idx:, :]

    return pd.concat([df_past, df])


def load_price_data():
    return load_raw_data()


def load_inv_price_data(inv_dummy_diff):
    df_inv = load_raw_data()
    df_inv['datetime'] = df_inv.index - datetime.timedelta(days=inv_dummy_diff)
    df_inv.set_index('datetime', inplace=True)
    return df_inv.iloc[::-1]


def jpy_to_usd(df, list):
    func = lambda x: x / 106.8
    for l in list:
        df[l] = df[l].apply(func)
    return df
