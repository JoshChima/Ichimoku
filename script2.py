import pandas as pd
import numpy as np
#from pandas_datareader import data, wb
import matplotlib as mpl
#from mpl_finance import candlestick_ohlc
import matplotlib.dates as dates, time
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
def MetaTraderDataConverter(file):
    #df=pd.read_csv("USD_JPY_2017_to_2018.csv")
    df=pd.read_csv(file, parse_dates=[['Date','Time']], sep='\t')
    df['Date'] = df['Date_Time']
    #reverse dataframe
    #df = df.iloc[::-1]
    #set date as index
    #df.set_index(df.Date, drop=True, inplace=True)
    df = df[['Date', 'Open', 'High', 'Low','Close']]
    return df


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
    d['pct_change'] = (d['Close']-d['Open'].shift(1))/d['Open'].shift(1)*100

    
    d.dropna(inplace=True)
    
    return d
#############################################
from math import sqrt
import pandas as pd
import numpy as np
import time
from sklearn import preprocessing
from collections import deque
import random
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
from keras import backend as K
K.tensorflow_backend._get_available_gpus()

datafile="USDJPY_H1_2014_2018.csv"
df = MetaTraderDataConverter(datafile)
df = df[['Date','Open', 'High', 'Low','Close']]
df.head()

df = Ichimoku(df)
#df['Date'] =  pd.to_datetime(df['Date'], format='%d%b%Y:%H:%M:%S.%f')
df = df.set_index(df['Date'])
df = df.drop('Date', axis=1)
df.head()

SEQ_LEN = 50
FUTURE_PERIOD_PREDICT = 3
RATIO_TO_PREDICT = "pct_change"
EPOCHS = 10
BATCH_SIZE = 64
NAME = str(SEQ_LEN) + "-SEQ-" + str(FUTURE_PERIOD_PREDICT) + "-PRED-" + str(int(time.time()))


def classify(future_pcp):
    if float(future_pcp) > 0:
        return 1
    else:
        return 0

def preprocessing_df(df):
    df = df.drop("pct_change", axis=1)
    
    for col in df.columns:
        if col != "target":
            df[col] = df[col].pct_change()
            df.dropna(inplace=True)
            df[col] = preprocessing.scale(df[col].values)
    df.dropna(inplace=True)
    
    sequential_data = []
    prev_days = deque(maxlen=SEQ_LEN)
    
    for i in df.values:
        prev_days.append([n for n in i[:-1]])
        if len(prev_days) == SEQ_LEN:
            sequential_data.append([np.array(prev_days), i[-1]])
            
    random.shuffle(sequential_data)
    
    buys = []
    sells = []
    
    for seq, target in sequential_data:
        if target == 0:
            sells.append([seq, target])
        elif target == 1:
            buys.append([seq, target])
    
    random.shuffle(buys)
    random.shuffle(sells)
    
    lower = min(len(buys), len(sells))
    
    buys = buys[:lower]
    sells = sells[:lower]
    
    sequential_data = buys+sells
    random.shuffle(sequential_data)
    
    X = []
    y = []
    
    for seq, target in sequential_data:
        X.append(seq)
        y.append(target)
        
    return np.array(X),y

df['target'] = list(map(classify, df[RATIO_TO_PREDICT].shift(-1)))

times = sorted(df.index.values)
last_5pct = times[-int(0.05*len(times))]
print(last_5pct)

validation_df = df[(df.index >= last_5pct)]
df = df[(df.index < last_5pct)]

train_x, train_y = preprocessing_df(df)
validation_x, validation_y = preprocessing_df(validation_df)

print("train data:",len(train_x),"validation:",len(validation_x))
print("Dont buys:", train_y.count(0), "buys:", train_y.count(1))
print("VALIDATION Dont buys:", validation_y.count(0), "buys:", validation_y.count(1))

model = Sequential()
model.add(CuDNNLSTM(64, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(CuDNNLSTM(64, return_sequences=True))
model.add(Dropout(0.1))
model.add(BatchNormalization())

model.add(CuDNNLSTM(64))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(2, activation='softmax'))

model.summary()


opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

# Compile model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy']
)

tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

filepath = "RNN_Final-{epoch:02d}-{val_acc:.3f}"  # unique file name that will include the epoch and the validation acc for that epoch
checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')) # saves only the best ones

# Train model
history = model.fit(
    train_x, train_y,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(validation_x, validation_y),
    callbacks=[tensorboard, checkpoint],
    )
# Score model
score = model.evaluate(validation_x, validation_y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# Save model
model.save("models/{}".format(NAME))

#tensorboard --logdir=logs