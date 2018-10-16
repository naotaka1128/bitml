import math
import pandas as pd
import numpy as np
from sklearn import preprocessing
from IPython import embed
import random
import itertools
import talib as ta

NON_NORMALIZE_COLUMNS = ['open', 'high', 'low', 'close', 'volume', 'flag']


def generate_data(args, df, delay, tick, flag_threshold, df_original=None):
    df = fix_error_row(df)
    df = calc_flag(df, delay, flag_threshold)
    df = calc_technicals(args, df, tick)
    df = calc_volume_technicals(args, df, tick)
    df = clip_outlier(args, df)
    df, df_real_price = add_change_rate(args, df)
    df = normalize_technicals(df)
    df = drop_unnecessary_columns(df)
    df = move_obj_value_to_tail(df)
    df = delete_correlate_columns(args, df, df_original)
    return df, df_real_price


def delete_correlate_columns(args, df, df_original):
    if args['delete_correlate_columns']:
        if df_original is None:
            df_corr = df.iloc[:, :-1].corr()
            df_not_correlated = ~(df_corr.mask(np.tril(np.ones([len(df_corr)]*2, dtype=bool))).abs() > args['delete_correlate_columns']).any()
            un_corr_idx = df_not_correlated.loc[df_not_correlated[df_not_correlated.index] == True].index
            df_out = df.iloc[:, :-1][un_corr_idx]
            return pd.concat([df_out, df.iloc[:, -1]], axis=1)
        else:
            columns = list(df_original.columns)
            columns.remove('adf flag')
            return pd.concat([df[columns], df.iloc[:, -1]], axis=1)
    else:
        return df


def move_obj_value_to_tail(df):
    df['adf flag'] = df['flag'].copy()
    df.drop(['flag'], 1, inplace=True)
    return df


def drop_unnecessary_columns(df):
    df.drop(['open', 'high', 'low', 'close', 'volume'], 1, inplace=True)
    return df


def normalize_technicals(df):
    min_max_scaler = preprocessing.MinMaxScaler()

    for column_name, item in df.iteritems():
        if column_name in NON_NORMALIZE_COLUMNS:
            continue
        df[column_name] = min_max_scaler.fit_transform(df[column_name].values.reshape(-1,1))

    return df


def add_change_rate(args, df):
    df_real_price = df.copy()

    prev_close = df['close'].shift(1)
    df['close_change_rate'] = df['close']
    df['close_change_rate'] /= prev_close

    prev_volume = df['volume'].shift(1)
    df['volume_change_rate'] = df['volume']
    df['volume_change_rate'] /= prev_volume

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df, df_real_price


def clip_outlier(args, df):
    threshold = args['threshold_outlier']
    for column_name, item in df.iteritems():
        if column_name in NON_NORMALIZE_COLUMNS:
            continue
        lower, upper = np.percentile(df[column_name].values,
                                     [threshold, 100 - threshold])
        df[column_name] = df[column_name].clip(lower, upper)

    return df


def calc_volume_technicals(args, df, tick):
    tick_seconds = tick * 60

    ## MovingAverage(Daily)
    MA3  = df.volume.rolling(window=3*60//tick*24).mean()
    MA5  = df.volume.rolling(window=5*60//tick*24).mean()
    MA10 = df.volume.rolling(window=10*60//tick*24).mean()
    MA25 = df.volume.rolling(window=25*60//tick*24).mean()
    df['vol_3MA_diff_percent_day'] = (MA3 - df['volume']) / df['volume']
    df['vol_5MA_diff_percent_day'] = (MA5 - df['volume']) / df['volume']
    df['vol_10MA_diff_percent_day'] = (MA10 - df['volume']) / df['volume']
    df['vol_25MA_diff_percent_day'] = (MA25 - df['volume']) / df['volume']

    df['vol_3MA_5MA_diff_day'] = df['vol_3MA_diff_percent_day'] - df['vol_5MA_diff_percent_day']
    df['vol_5MA_10MA_diff_day'] = df['vol_5MA_diff_percent_day'] - df['vol_10MA_diff_percent_day']
    df['vol_5MA_25MA_diff_day'] = df['vol_5MA_diff_percent_day'] - df['vol_25MA_diff_percent_day']
    df.fillna(0, inplace=True)

    return df


def calc_technicals(args, df, tick):
    tick_seconds = tick * 60
    open = np.array(df.open, dtype='f8')
    high = np.array(df.high, dtype='f8')
    low = np.array(df.low, dtype='f8')
    close = np.array(df.close, dtype='f8')
    volume = np.array(df.volume, dtype='f8')

    day_of_week = df.index.map(lambda x: x.weekday())
    df['day_of_week_sin'] = day_of_week.map(lambda x: math.sin(math.radians(x/7*360)))
    df['day_of_week_cos'] = day_of_week.map(lambda x: math.cos(math.radians(x/7*360)))

    df['high_low'] = df['high'] - df['low']
    df['high_low'] /= df['close']
    df['open_low'] = df['open'] - df['low']
    df['open_low'] /= df['close']
    df['open_high'] = df['open'] - df['high']
    df['open_high'] /= df['close']

    #################### MovingAverage
    ## Daily
    for i in [3, 5, 10, 25, 75, 200]:
        sma = ta.SMA(close, timeperiod=i*60//tick*24)
        df[str(i) + 'MA_diff_percent_day'] = (sma - df['close']) / df['close']

    for i, j in itertools.combinations([3, 5, 10, 25], 2):
        new_column = str(i) + 'MA_' + str(j) + 'MA_diff_day'
        column_1 = str(i) + 'MA_diff_percent_day'
        column_2 = str(j) + 'MA_diff_percent_day'
        df[new_column] = df[column_1] - df[column_2]
        for k in range(1, 2):  # 3でも大丈夫かもだが大差ない
            new_column_2 = str(i) + 'MA_' + str(j) + 'MA_diff_day_before' + str(k)
            df[new_column_2] = df[new_column].shift(k) - df[new_column]
    df.fillna(0, inplace=True)

    #################### Momentum
    ## Momentum
    for i in [3, 5, 10, 20]:
        Mom_period = i * 60 // tick * 24
        shift = df.close.shift(Mom_period)
        df['Mom' + str(i)] = df.close / shift * 100


    #################### Bolinger / MACD
    for i in [3, 9, 20, 25, 50]:
        base = df.close.rolling(window=i*60//tick*24).mean()
        sigma = df.close.rolling(window=i*60//tick*24).std(ddof=0)
        upper_1sigma = base + 1 * sigma
        lower_1sigma = base - 1 * sigma
        upper_2sigma = base + 2 * sigma
        lower_2sigma = base - 2 * sigma
        df[str(i) + 'MAbb_upper_1sigma_diff_percent'] = (upper_1sigma - df['close']) / df['close']
        df[str(i) + 'MAbb_lower_1sigma_diff_percent'] = (lower_1sigma - df['close']) / df['close']
        df[str(i) + 'MAbb_upper_2sigma_diff_percent'] = (upper_2sigma - df['close']) / df['close']
        df[str(i) + 'MAbb_lower_2sigma_diff_percent'] = (lower_2sigma - df['close']) / df['close']

    ## MACD
    FastEMA_period = 12*60//tick*24
    SlowEMA_period = 26*60//tick*24
    SignalSMA_period = 9*60//tick*24
    df['MACD'] = df.close.ewm(span=FastEMA_period).mean() - df.close.ewm(span=SlowEMA_period).mean()
    df['Signal'] = df['MACD'].rolling(window=SignalSMA_period).mean()
    df['MACD_Signal_diff'] = df['MACD'] - df['Signal']

    #################### RSI / HLBand / Stoch
    ## RSI
    for i in [5, 10, 14, 30]:
        RSI_period = i*60//tick*24
        one_day_before = 1*60//tick*24
        diff = df.close.diff(one_day_before)
        positive = diff.clip_lower(0).ewm(alpha=1/RSI_period).mean()
        negative = diff.clip_upper(0).ewm(alpha=1/RSI_period).mean()
        df['RSI' + str(i)] = 100 - 100 / (1 - positive / negative)

    ## HLband
    for i in [3, 7, 20, 40]:
        period = i*60//tick*24
        Hline = df.close.rolling(period).max()
        Lline = df.close.rolling(period).min()
        df['Hline_diff_percent_' + str(i)] = (Hline - df['close']) / df['close']
        df['Lline_diff_percent_' + str(i)] = (Lline - df['close']) / df['close']

    df.fillna(0, inplace=True)

    ## Stochastics
    Kperiod = 14*60//tick*24  # %K
    Dperiod = 3*60//tick*24   # %D
    Slowing = 3*60//tick*24
    Hline = df.high.rolling(Kperiod).max()
    Lline = df.low.rolling(Kperiod).min()
    sumlow = (df.close - Lline).rolling(Slowing).sum()
    sumhigh = (Hline - Lline).rolling(Slowing).sum()
    df['Stoch_day'] = sumlow / sumhigh * 100
    df['StochSignal_day'] = df['Stoch_day'].rolling(Dperiod).mean()
    df.fillna(0, inplace=True)

    #################### VIX / Ichimoku

    # VIX
    import collections
    def vixfix(close, low, high, period=22, bbl=20, mult=2.0, lb=50, ph=0.85, pl=1.01):
        period = period  # LookBack Period Standard Deviation High
        bbl = bbl  # Bolinger Band Length
        mult = mult  # Bollinger Band Standard Devaition Up
        lb = lb  # Look Back Period Percentile High
        ph = ph  # Highest Percentile - 0.90=90%, 0.95=95%, 0.99=99%
        pl = pl  # Lowest Percentile - 1.10=90%, 1.05=95%, 1.01=99%
        hp = False  # Show High Range - Based on Percentile and LookBack Period?
        sd = False  # Show Standard Deviation Line?
        # VixFix
        wvf = (close.rolling(period, 1).max() - low) / close.rolling(period, 1).max() * 100
        # VixFix_inverse
        wvf_inv = abs((close.rolling(period, 1).min() - high) / close.rolling(period, 1).min() * 100)
        sDev = mult * pd.Series(wvf).rolling(bbl, 1).std()
        midLine = pd.Series(wvf).rolling(bbl, 1).mean()
        lowerBand = midLine - sDev
        upperBand = midLine + sDev
        rangeHigh = pd.Series(wvf).rolling(lb, 1).max() * ph
        rangeLow = pd.Series(wvf).rolling(lb, 1).min() * pl
        result = collections.namedtuple('result', 'wvf, wvf_inv, lowerBand, upperBand, rangeHigh,rangeLow')
        return result(wvf=wvf, wvf_inv=wvf_inv, lowerBand=lowerBand, upperBand=upperBand,
                    rangeHigh=rangeHigh, rangeLow=rangeLow)

    vix = vixfix(close=df.close, low=df.low, high=df.high)
    df['vix_wvf'] = vix.wvf
    df['vix_wvf_inv'] = vix.wvf_inv

    # Ichimoku
    band9_period = 9*24*60*60 // tick_seconds
    Hline9 = df.close.rolling(band9_period).max()
    Lline9 = df.close.rolling(band9_period).min()
    change_line = ( Hline9 + Lline9 ) / 2

    band26_period = 26*24*60*60 // tick_seconds
    Hline26 = df.close.rolling(band26_period).max()
    Lline26 = df.close.rolling(band26_period).min()
    standart_line = ( Hline26 + Lline26 ) / 2

    span1_line = (change_line + standart_line) / 2

    band52_period = 52*24*60*60 // tick_seconds
    Hline52 = df.close.rolling(band52_period).max()
    Lline52 = df.close.rolling(band52_period).min()
    span2_line = ( Hline52 + Lline52 ) / 2

    df['Ichimoku_change_line_diff_percent'] = (change_line - df['close']) / df['close']
    df['Ichimoku_standart_line_diff_percent'] = (standart_line - df['close']) / df['close']
    df['Ichimoku_span1_line_diff_percent'] = (span1_line - df['close']) / df['close']
    df['Ichimoku_span2_line_diff_percent'] = (span2_line - df['close']) / df['close']

    #################### TA-Lib params (Volatility)
    df['NATR_1day'] = ta.NATR(high, low, close, timeperiod=14*60//tick*24)
    for i in [1, 2]:
        new_column_name = 'NATR_' + str(i) + 'days_before_diff'
        df[new_column_name] = df['NATR_1day'] - df['NATR_1day'].shift(i*60//tick*24)

    #################### TA-Lib params (Mom系)
    adxr = ta.ADXR(high, low, close, timeperiod=14*60//tick*24)
    adx = ta.ADX(high, low, close, timeperiod=14*60//tick*24)
    df['ADX_ADXR_diff'] = adx - adxr

    #################### TA-Lib params (Other)
    df['HT_SINE'], df['leadsine'] = ta.HT_SINE(close)
    df['BETA'] = ta.BETA(high, low, timeperiod=5*60//tick*24)
    df['CORREL'] = ta.CORREL(high, low, timeperiod=30*60//tick*24)

    #################### RCI
    def ord(seq, idx, itv):
        p = seq[idx]
        o = 1
        for i in range(0, itv):
            if p < seq[i]:
                o = o + 1
        return o

    def d(itv, src):
        sum = 0.0
        for i in range(0, itv):
            sum = sum + pow((i + 1) - ord(src, i, itv), 2)
        return sum

    def calc_rci(itv, src):
        rciM = (1.0 - 6.0 * d(itv,src) / (itv * (itv * itv - 1.0))) * 100.0
        return rciM

    def get_rci(period_rci, close):
        rank_period = np.arange(period_rci, 0, -1)
        close_len = len(close)
        rci = np.zeros(close_len)
        for i in range(close_len - period_rci + 1):
            rci[-i-1] = calc_rci(period_rci, close[close_len - period_rci - i : close_len - i])
        return rci

    for period in [9, 36]:
        df['RCI_' + str(period)] = get_rci(period, df.close.tolist())

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    return df


def calc_flag(df, delay, flag_threshold):
    def check_change(changeHigh_series, changeLow_series):
        change = []
        changeHigh, changeLow = changeHigh_series.tolist(), changeLow_series.tolist()
        for i in range(len(changeHigh)):
            if changeHigh[i] >= (1 + flag_threshold) and \
               changeLow[i] < (1 - flag_threshold):
                if (changeHigh[i] - 1) >= (1 - changeLow[i]):
                    change.append(2)
                else:
                    change.append(1)
            elif changeHigh[i] >= (1 + flag_threshold):
                change.append(2)
            elif changeLow[i] < (1 - flag_threshold):
                change.append(1)
            else:
                change.append(0)
        return change

    rollingHigh = df.high.rolling(window=delay).max()
    rollingLow = df.low.rolling(window=delay).min()
    changeHigh = rollingHigh.shift(-1*delay)
    changeLow = rollingLow.shift(-1*delay)
    changeHigh /= df['close']
    changeLow /= df['close']

    df['flag'] = check_change(changeHigh, changeLow)
    df.dropna(inplace=True)
    return df


def fix_error_row(df):
    df.drop(df[df['low'] < 20].index, axis=0, inplace=True)
    return df
