import rqdatac as rq
from dotenv import load_dotenv
import os

import pandas as pd
import numpy as np
import datetime
from scipy.stats import norm
import math

from matplotlib import pyplot as plt
from pprint import pprint

start_time = datetime.datetime.now()

# Initialize rqdatac, source of financial data
base_dir = os.path.dirname(os.path.realpath(__file__))
load_dotenv()
token_path = os.getenv("RQDATAC2_CONF")
os.environ['RQDATAC2_CONF'] = token_path
rq.init()


class _g():
    pass

G = _g()

class Bucket:
    def __init__(self, data):
        self.data = data

    def std_deviation(self):
        # Find standard deviation
        return np.std(self.data.Delta_P)
    
    def add_volBS_est(self):
        z = self.data.Delta_P/self.std_deviation()
        self.data['volBuy'] = self.data.vol*norm.cdf(z, 0, 1)
        self.data['volSell'] = self.data.vol*(1-norm.cdf(z, 0, 1))
        return self.data

    def show_heads(self):
        return self.data.iloc[:5]

def load_data(id, date_start, date_end):
    try:
        prices_df = pd.read_csv(base_dir+f'/VPIN_Example.csv', index_col=True)
    except:
        prices_df = rq.get_price(id, start_date=date_start, end_date=date_end, fields=['last', 'volume'],frequency='tick')
        prices_df = prices_df.reset_index().drop('order_book_id', axis=1)
        prices_df.to_csv(base_dir+f'/VPIN_{id}_{date_start}_{date_end}.csv')
    return prices_df

def get_avg_volume(id, date_start, date_end, n=20):
    dailyVol_df = rq.get_price(id, start_date=date_start, end_date=date_end, fields=['volume'],frequency='1d')
    dailyVol_df = dailyVol_df.reset_index().drop('order_book_id', axis=1)
    return int(round(np.mean(dailyVol_df.volume)/(4*n),0))

def generate_bucket(dataframe, low, up):
    bucket = dataframe[dataframe.volCum>low]
    bucket = bucket[bucket.volCum<up]
    bucket = pd.concat([bucket, dataframe[dataframe.index==bucket.index[-1]+1]])
    a, b = bucket.volCum.iloc[0]-low, up - bucket.volCum.iloc[-2]
    bucket.at[bucket.index[0], 'vol'] = a
    bucket.at[bucket.index[-1], 'vol'] = b
    bucket.at[bucket.index[-1], 'volCum'] = up
    return bucket

def get_ecdf(vpin_array, method='percentile'):
    """
    cdf_method can be "percentile" to rank the VPIN values find their percentiles,
    or it can be "normal" to find its z-score and convert it to a CDF of a normal distrubution.
    """
    vpin_current = vpin_array.iloc[-1]
    ecdf_df = pd.DataFrame()
    ecdf_df['vpin_sorted'] = np.sort(vpin_array)
    if method == 'percentile':
        ecdf_df['ecdf'] = np.arange(1, len(ecdf_df.vpin_sorted)+1)/len(ecdf_df.vpin_sorted)
        return ecdf_df[ecdf_df.vpin_sorted==vpin_current].ecdf.values[0]
    elif method == 'normal':
        sample_mean = np.mean(ecdf_df.vpin_sorted)
        sample_std = np.std(ecdf_df.vpin_sorted)
        z_score = (vpin_current - sample_mean)/sample_std
        return norm.cdf(z_score/2)

def handle_data(symbol, start_date, end_date):
    handle_data_df = load_data(symbol, start_date, end_date)
    handle_data_df = handle_data_df.rename(columns = {'last':'p_i', 'volume':'vol_intradayCum'})
    handle_data_df['Delta_P'] = handle_data_df.p_i - handle_data_df.p_i.shift(1)
    handle_data_df['vol'] = handle_data_df.vol_intradayCum - handle_data_df.vol_intradayCum.shift(1)
    handle_data_df['vol'] = [max(0.0, v) for v in handle_data_df.vol]
    handle_data_df['volCum'] = handle_data_df.vol.cumsum()
    return handle_data_df

def generate_bucket_list(prices_df, k=12):
    """
    Find an bucket division method so that each bucket contains at least 40 trading records.
    We take the higher one between the average volume of each bucket and
    the maximum volume (in practice, maximum*1.001) over the entire period as the bucket volume.
    If the "maximum volume" result violates the first criteria (trading per bucket > 39), 
    we give up the current division and move on to the next until we reach an "average volume"
    higher than the "maximum volume". Then we find the maximum divisor k to meet the first criteria.
    When lower the divisor k, we are widening the bucket size until it reaches the criteria above.
    e.g. k=60 ~ 1 minute, k=12 ~ 5 minutes, k=1 ~ 60 minutes, k=0.5 ~ 120 minutes, etc.
    """
    max_adpted_signal = 0
    max_fail_signal = 0
    while k>0:
        minute, second = divmod(60/k, 1)
        print(k, f"- average bucket volume equal to a time volume of {int(minute)}'{int(second*60)}'':")
        G.threshold = get_avg_volume(G.symbol, G.start, G.end, k)
        print('V_avg =',G.threshold)
        G.volMax = np.max(prices_df.vol)
        print('V_max =', G.volMax)
        if G.threshold < G.volMax:
            G.threshold = G.volMax*1.0001 # Add a multiplier in case the volume of first trade is the largest.
            max_adpted_signal = 1
            print('Maximum adopted')
        else:
            max_adpted_signal = 0
            print('Average adopted')
        if max_adpted_signal * max_fail_signal == 0:
            thresholds_list = [i*G.threshold for i in range(int(np.max(prices_df.volCum)/G.threshold)+1)]
            buckets_list = []
            i = 0
            while i < len(thresholds_list)-1:
            #for i in range(len(thresholds_list)-1):
                low_bound, up_bound = thresholds_list[i], thresholds_list[i+1]
                bucket_df = generate_bucket(prices_df, low_bound, up_bound)
                if bucket_df.shape[0] > 39:
                    bucket_instance = Bucket(bucket_df)
                    buckets_list.append(bucket_instance)
                    i += 1
                else:
                    print("Some bucket contains # trades =", bucket_df.shape[0], ", which is too low.")
                    buckets_list = []
                    if max_adpted_signal == 1:
                        max_fail_signal = 1
                    if (k>0) and (k<1):
                        k = k/2
                    else:
                        k -= 1
                        if k == 0:
                            k = 0.5
                    break
            if len(buckets_list) != 0:
                G.buckets_data_list = [bk.add_volBS_est() for bk in buckets_list]
                G.tradesPerBucket_min = np.min([vdf.shape[0] for vdf in G.buckets_data_list])
                break
        else:
            if (k>0) and (k<1):
                k = k/2
            else:
                k -= 1
                if k == 0:
                    k = 0.5

    print("k =", k)
    print("Minimum Trades per bucket =", G.tradesPerBucket_min)
    print('V =', G.threshold)
    print('n =', len(G.buckets_data_list))
    print(G.buckets_data_list[0])
    print(G.buckets_data_list[-1])
    return thresholds_list

def get_vpin(thresholds_list, cdf_method):
    """
    Calculate individual VPIN given the data.
    cdf_method can be "percentile" to rank the VPIN values find their percentiles,
    or it can be "normal" to find its z-score and convert it to a CDF of a normal distrubution.
    """
    V_df = pd.DataFrame()
    V_df['V_cum'] = thresholds_list[1:]
    V_df['close_time'] = [data.iloc[-1].datetime for data in G.buckets_data_list]
    V_df['Vtau_Buy'] = [np.sum(data.volBuy) for data in G.buckets_data_list]
    V_df['Vtau_Sell'] = [np.sum(data.volSell) for data in G.buckets_data_list]
    V_df['abs_dif'] = abs(V_df.Vtau_Buy - V_df.Vtau_Sell)
    V_df['VPIN'] = V_df.abs_dif.rolling(window=int(V_df.shape[0]/4)).mean()/(G.threshold)
    V_df = V_df.dropna()
    V_ecdf_list = []
    for idx in V_df.index:
        window = int(V_df.shape[0]/4)
        start_idx = idx - window + 1
        VPIN_field = V_df.loc[start_idx:idx,:].VPIN
        if VPIN_field.shape[0] == window:
            ecdf_value = get_ecdf(VPIN_field, method=cdf_method)
        else:
            ecdf_value = None
        V_ecdf_list.append(ecdf_value)
    V_df['ECDF'] = V_ecdf_list
    V_df = V_df.dropna().reset_index().drop('index', axis=1)
    V_df['x_tick'] = V_df.close_time.astype('str')
    for idx in V_df.index:
        if idx % int(V_df.shape[0]/9) != 0:
            V_df.loc[idx,'x_tick'] = ''
    print(V_df)
    return V_df

def generate_plot(V_df):
    plt.plot(V_df.index, V_df.VPIN, label='VPIN')
    plt.plot(V_df.index, V_df.ECDF, label='VPIN(CDF)')
    plt.xticks(V_df.index, V_df.x_tick, rotation=30)
    plt.legend()
    plt.title(f'VPIN-Relative VPIN(CDF): {G.symbol}\n\
        {V_df.close_time.iloc[0]}\ ~ {V_df.close_time.iloc[-1]}\n\
        Last VPIN: {V_df.VPIN.iloc[-1]}, Last VPIN(CDF): {V_df.ECDF.iloc[-1]}')
    plt.show()

def main(symbol, start_date, end_date, k_init=12, cdf_method='percentile', to_plot=False):
    prices_df = handle_data(symbol, start_date, end_date)
    # print(prices_df)
    print("Finding an appropirate bucket size:")
    # Set up V
    thresholds_list = generate_bucket_list(prices_df, k_init)
    # Compute VPIN from buckets
    V_df = get_vpin(thresholds_list, cdf_method)
    print(f'{datetime.datetime.now()-start_time}: Mission completed')
    if to_plot==True:
        generate_plot(V_df)

if __name__ == '__main__':
    print(f'{datetime.datetime.now()-start_time}: Environment initialized')
    """
    Common stock - large market value
    """
    #G.symbol = '600519.XSHG' # 茅台
    #G.symbol = '002594.XSHE' # 比亚迪
    """
    Common stock - smaller market value
    """
    #G.symbol = '603009.XSHG' # 北特科技
    #G.symbol = '300736.XSHE' # 百邦科技
    """
    Index futures
    """
    #G.symbol = 'IF888' # 沪深300期货
    #G.symbol = 'IC888' # 中证500期货
    """
    Commodity futures
    """
    #G.symbol = 'JM888' # 焦煤期货
    G.symbol = 'RB888' # 螺纹钢期货

    G.start = '2024-02-01'
    G.end = '2024-08-02'

    main(G.symbol, G.start, G.end, k_init=60, cdf_method = 'normal', to_plot=True)
