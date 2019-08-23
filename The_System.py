import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Dense, Activation
from keras.models import Sequential, load_model
from keras.optimizers import Adam

from Methods import MetaTraderDataConverter, Ichimoku
datafile="USDJPY_H1_2014_2018.csv"


class order:
    def __init__(self, order_number, current_date, current_price, lot_size, is_buy):
        self.ORDER_NUMBER = order_number
        self.current_date = current_date
        self.starting_date = current_date

        self.is_buy = is_buy
        self.is_open = True

        self.starting_price = current_price
        self.current_price = current_price
        self.lot_size = lot_size
        self.pip = 0.0000
        self.profit = 0

    def PIP(self, isBuy):
        if isBuy:
            pip = float(self.current_price - self.starting_price) * 100
            return pip
        else:
            pip = float(self.starting_price - self.current_price) * 100
            return pip
        
    def update(self, current_price, current_date):
        self.current_price = current_price
        self.current_date = current_date
        self.pip = self.PIP(self.is_buy)
        self.profit = self.pip * self.lot_size
    def get_profit(self):
        return self.profit
    def get_order_number(self):
        return self.ORDER_NUMBER
    def close(self):
        self.is_open = False

class Mainframe:
    def __init__(self, data, rng, balance):
        self.dataframe = MetaTraderDataConverter(data)

        self.balance = balance
        self.profit = 0

        self.current_t = rng
        self.range = rng
        self.furthest_t = self.current_t - self.range

        self.DF_SUBSET = Ichimoku(self.dataframe.iloc[self.furthest_t:self.current_t+1])
        # self.highs = [self.dataframe['High'].iloc[self.furthest_t:self.current_t+1]]
        # self.opens = [self.dataframe['Open'].iloc[self.furthest_t:self.current_t+1]]
        # self.lows = [self.dataframe['Low'].iloc[self.furthest_t:self.current_t+1]]
        # self.closes = [self.dataframe['Close'].iloc[self.furthest_t:self.current_t+1]]
        
        self.date_t = self.dataframe.index[self.current_t]
        self.price_t = self.dataframe['Close'].iloc[self.current_t]
        
        self.NUMBER_OF_TRADES = 0
        self.lot_size = 0.1
        self.OPEN_TRADES = []
        self.TRADE_HISTORY = {}
        
        #self.stateSpace = [self.balance, self.profit, self.current_t, self.]

    def forward(self):
        self.current_t +=1
        self.furthest_t = self.current_t - self.range
        self.DF_SUBSET = Ichimoku(self.dataframe.iloc[self.furthest_t:self.current_t+1])
        self.date_t = self.dataframe.index[self.current_t]
        self.price_t = self.dataframe['Close'].iloc[self.current_t]
        for order in self.OPEN_TRADES:
            order.update(self.price_t, self.date_t)
        self.profits()

    def buy(self):
        self.show()
        if len(self.OPEN_TRADES) < 1:
            self.NUMBER_OF_TRADES +=1
            ORDER = order(self.NUMBER_OF_TRADES, self.date_t,self.price_t,self.lot_size,True)
            self.OPEN_TRADES.append(ORDER)
            self.TRADE_HISTORY['{}'.format(self.NUMBER_OF_TRADES)] = ORDER
            self.forward()
        else:
            "Trade Limit reached"
    def sell(self):
        self.show()
        if len(self.OPEN_TRADES) < 1:
            self.NUMBER_OF_TRADES +=1
            ORDER = order(self.NUMBER_OF_TRADES, self.date_t,self.price_t,self.lot_size,False)
            self.OPEN_TRADES.append(ORDER)
            self.TRADE_HISTORY['{}'.format(self.NUMBER_OF_TRADES)] = ORDER
            self.forward()
        else:
            "Trade Limit reached"
    def closeTrade(self,choice):
        self.show()
        choices = [c for c in range(0,len(self.OPEN_TRADES))]
        if choice in choices:
            ORDER = self.OPEN_TRADES[choice]
            ORDER.close()
            self.OPEN_TRADES.pop(choice)
            self.TRADE_HISTORY['{}'.format(ORDER.get_order_number())] = ORDER

    def profits(self):
        total_profit = 0
        already_checked = []
        for ordernum in self.TRADE_HISTORY.keys():
            if ordernum not in already_checked:
                ORDER = self.TRADE_HISTORY[ordernum]
                total_profit += ORDER.get_profit()
                already_checked.append(ordernum)
        for order in self.OPEN_TRADES:
            if order.get_order_number() not in already_checked:
                total_profit += order.get_profit()
                already_checked.append(order.get_order_number())
        self.profit += total_profit

    def show(self):
        print("##################################")
        print("Profit: {}".format(self.profit))
        print("Balance: {}".format(self.balance + self.profit))
        print(self.dataframe.index[self.furthest_t], 'to', self.dataframe.index[self.current_t])
    
     #def action(self, choice):
      #   if choice == 0:

Trader = Mainframe(datafile, 100, 10000)

Trader.show()
for i in range(5):
    Trader.forward()
Trader.sell()
for i in range(50):
    Trader.forward()
Trader.closeTrade(0)
for i in range(5):
    Trader.forward()
Trader.buy()
for i in range(50):
    Trader.forward()
Trader.closeTrade(0)

# df = MetaTraderDataConverter(datafile)
# #print(df.head())
# #print(df['Close'].iloc[10] - df['Close'].iloc[5])
# #print(df.index[2])
# print(df.index[3])