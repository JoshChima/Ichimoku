import pandas as pd
from pandas_datareader import data, wb
import matplotlib as mpl
from mpl_finance import candlestick_ohlc
import matplotlib.dates as dates, time
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import quandl
quandl.ApiConfig.api_key = "ixzQFtzftvnjuDSrLW9h"

d=pd.read_csv("USD_JPY_2017_to_2018.csv")

#reverse dataframe
d = d.iloc[::-1]

#calculate close price
d["Close"] = d.shift(-1)["Open"]

#set date as index
d.set_index(d.Date, drop=True, inplace=True)

#print(d.head())
#print(d.shape)
#print(d.iloc[:5])

def df_slices(dataframe):
    sliced_frames = []
    for shift in range(60,dataframe.shape[0]):
        df_slice = dataframe.iloc[:shift]
        sliced_frames.append(df_slice)
    return sliced_frames

def Ichimoku(dataframe):
    d = dataframe
    nine_period_high = d['High'].rolling(window= 9).max()
    nine_period_low = d['Low'].rolling(window= 9).min()
    d['tenkan_sen'] = (nine_period_high + nine_period_low) /2
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2))
    period26_high = d['High'].rolling(window=26).max()
    period26_low = d['Low'].rolling(window=26).min()
    d['kijun_sen'] = (period26_high + period26_low) / 2
    # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2))
    d['senkou_span_a'] = ((d['tenkan_sen'] + d['kijun_sen']) / 2).shift(26)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2))
    period52_high = d['High'].rolling(window=52).max()
    period52_low = d['Low'].rolling(window=52).min()
    d['senkou_span_b'] = ((period52_high + period52_low) / 2).shift(52)
    # The most current closing price plotted 26 time periods behind (optional)
    d['chikou_span'] = d['Close'].shift(-26)
    return d

sf = df_slices(d)

#print(Ichimoku(sf[300]))
print(Ichimoku(d))
#for s in range(len(sf)):
 #   print("#%s"%(s))
  #  print("Start: ",sf[s].iloc[0])
   # print("End: ",sf[s].iloc[sf[s].__len__()-1])
    #print("###########")