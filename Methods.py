import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import plotly
#import plotly.plotly as py
import plotly.graph_objs as go
import os
from plotly.offline import init_notebook_mode, plot, iplot
import plotly

def MetaTraderDataConverter(file):
    df=pd.read_csv(file, parse_dates=[['Date','Time']], sep='\t')
    df['Date'] = df['Date_Time']
    df.set_index(df.Date, drop=True, inplace=True)
    df = df[['Open', 'High', 'Low','Close']]
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
    d['pct_change'] = (d['Close']-d['Close'].shift(1))/d['Close'].shift(1)*100

        
    return d

def Ichimoku_plot(d):
    # Set colours for up and down candles
    INCREASING_COLOR = 'green'
    DECREASING_COLOR = 'red'
    # create list to hold dictionary with data for our first series to plot
    # (which is the candlestick element itself)
    data1 = [ dict(
        type = 'candlestick',
        open = d.Open,
        high = d.High,
        low = d.Low,
        close = d.Close,
        x = d.index,
        yaxis = 'y2',
        name = 'F',
        increasing = dict( line = dict( color = INCREASING_COLOR ) ),
        decreasing = dict( line = dict( color = DECREASING_COLOR ) ),
    ) ]
    # Create empty dictionary for later use to hold settings and layout options
    layout=dict()
    # create our main chart "Figure" object which consists of data to plot and layout settings
    fig = dict( data=data1, layout=layout )
    # Assign various seeting and choices - background colour, range selector etc
    fig['layout']['plot_bgcolor'] = 'grey'
    fig['layout']['xaxis'] = dict( rangeselector = dict( visible = True ) )
    fig['layout']['yaxis'] = dict( domain = [0, 0.2], showticklabels = False )
    fig['layout']['yaxis2'] = dict( domain = [0.2, 0.8] )
    fig['layout']['legend'] = dict( orientation = 'h', y=0.9, x=0.3, yanchor='bottom' )
    fig['layout']['margin'] = dict( t=40, b=40, r=40, l=40 )
    # Populate the "rangeselector" object with necessary settings
    rangeselector=dict(
        visible = True,
        x = 0, y = 0.9,
        bgcolor = 'rgba(150, 200, 250, 0.4)',
        font = dict( size = 13 ),
        buttons=list([
            dict(count=1,
                 label='reset',
                 step='all'),
            dict(count=1,
                 label='1yr',
                 step='year',
                 stepmode='backward'),
            dict(count=3,
                label='3 mo',
                step='month',
                stepmode='backward'),
            dict(count=1,
                label='1 mo',
                step='month',
                stepmode='backward'),
            dict(step='all')
        ]))

    fig['layout']['xaxis']['rangeselector'] = rangeselector
    # Append the Ichimoku elements to the plot
    fig['data'].append( dict( x=d['tenkan_sen'].index, y=d['tenkan_sen'], type='scatter', mode='lines', 
                             line = dict( width = 1 ),
                             marker = dict( color = '#e7e14f' ),
                             yaxis = 'y2', name='tenkan_sen' ) )
    fig['data'].append( dict( x=d['kijun_sen'].index, y=d['kijun_sen'], type='scatter', mode='lines', 
                             line = dict( width = 1 ),
                             marker = dict( color = '#20A4F3' ),
                             yaxis = 'y2', name='kijun_sen' ) )
    fig['data'].append( dict( x=d['senkou_span_a'].index, y=d['senkou_span_a'], type='scatter', mode='lines', 
                             line = dict( width = 1 ), 
                             marker = dict( color = '#228B22' ),
                             yaxis = 'y2', name='senkou_span_a' ) )
    fig['data'].append( dict( x=d['senkou_span_b'].index, y=d['senkou_span_b'], type='scatter', mode='lines', 
                             line = dict( width = 1 ),fill='tonexty',
                             marker = dict( color = '#FF3342' ),
                             yaxis = 'y2', name='senkou_span_b' ) )
    fig['data'].append( dict( x=d['chikou_span'].index, y=d['chikou_span'], type='scatter', mode='lines', 
                             line = dict( width = 1 ),
                             marker = dict( color = '#D105F5' ),
                             yaxis = 'y2', name='chikou_span' ) )
    
    # Set colour list for candlesticks
    colors = []
    for i in range(len(d.Close)):
        if i != 0:
            if d.Close[i] > d.Close[i-1]:
                colors.append(INCREASING_COLOR)
            else:
                colors.append(DECREASING_COLOR)
        else:
            colors.append(DECREASING_COLOR)
    
    if not os.path.exists("images"):
        os.mkdir("images")
    
    dload = os.path.expanduser('./images')
    html_file = 'candlestick-ichimoku.html'
    fname = 'candlestick-ichimoku'


    iplot(fig, filename="candlestick-ichimoku")
    return d
    #fig.show(renderer="png")

def u_d(num):
    if num < 0:
        return -1
    else:
        return 1

def NeoCandle(OHLC):
    
    df = OHLC
    df['Change'] = df['Close']-df['Close'].shift(1)
    df['U_D'] = df['Change'].apply(u_d)
    df['DHL'] = (df['High'] - df['Low'])
    df['DOC'] = abs(df['Open'] - df['Close']) / (df['High'] - df['Low']) #percentage of space taken in DHL by DOC

    df_OC = df[['Open','Close']]
    # df_OC['Mid'] = df_OC.median(axis=1)
    df_OC['Max'] = df_OC.max(axis=1)
    # df['PODD'] = df_OC['Mid']
    df['PODD'] = ((df_OC['Max'] - (abs(df['Open'] - df['Close'])/2)) - df['Low']) / (df['High'] - df['Low'])
    #for i in range(df.shape[0]):
     #   df['PODD'].iloc[i] = (df_OC['Mid'].iloc[i] / df_OC['Max'].iloc[i] - 0.999) * 1000
    return df

def PODD(OPEN, HIGH, LOW, CLOSE):
    Max = max([OPEN, CLOSE])
    DOChalf = abs(OPEN - CLOSE) / 2
    DHLmini = HIGH - LOW
    print((Max - DOChalf - LOW) / DHLmini)

#PODD(102.748, 102.839, 102.688, 102.791)
