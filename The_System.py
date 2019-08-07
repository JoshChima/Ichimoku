import pandas as pd

datafile="USDJPY_H1_2014_2018.csv"
def MetaTraderDataConverter(file):
    df=pd.read_csv(file, parse_dates=[['Date','Time']], sep='\t')
    df['Date'] = df['Date_Time']
    df.set_index(df.Date, drop=True, inplace=True)
    df = df[['Open', 'High', 'Low','Close']]
    return df
df = MetaTraderDataConverter(datafile)
print(df.head())
print(df.iloc[5])
class order:
    def __init__(self, current_date, current_price, lot_size, is_buy):
        self.current_date = current_date
        self.starting_date = current_date
        self.is_buy = is_buy
        self.starting_price = current_price
        self.current_price = current_price
        self.lot_size = lot_size
        self.pip = pip(self.is_buy)
        self.profit = self.pip * self.lot_size

    def pip(self, is_buy):
        if is_buy:
            pip = (self.current_price - self.starting_price) * 100
        else:
            pip = (self.starting_price - self.current_price) * 100
        return pip
    def update(self, current_price, current_date):
        self.current_price = current_price
        self.current_date = current_date
class Mainframe:
    def __init__(self, data, rng):
        self.dataframe = MetaTraderDataConverter(data)

        self.current_t = rng
        self.range = rng
        self.furthest_t = self.current_t - self.range

        self.highs = [self.dataframe['High'].iloc[self.furthest_t:self.current_t+1]]
        self.opens = [self.dataframe['Open'].iloc[self.furthest_t:self.current_t+1]]
        self.lows = [self.dataframe['Low'].iloc[self.furthest_t:self.current_t+1]]
        self.closes = [self.dataframe['Close'].iloc[self.furthest_t:self.current_t+1]]
        
        self.date_t = self.dataframe.index.iloc[self.current_t]
        self.price_t = self.dataframe['Close'].iloc[self.current_t]
        
        self.lot_size = 0.01
        self.OPEN_TRADES = []
    def forward(self):
        self.current_t +=1
    def buy(self):
        ORDER = order(self.date_t,self.price_t,self.lot_size,True)
        self.OPEN_TRADES.append(ORDER)
    def sell(self):
        ORDER = order(self.date_t,self.price_t,self.lot_size,False)
        self.OPEN_TRADES.append(ORDER)




    # def action(self, choice):
    #     if choice == 1:



