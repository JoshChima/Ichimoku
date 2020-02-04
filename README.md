# Ichimoku
 Predicting %change in price of USD/JPY currency pair with a LSTM

This Project is not ready for use in real market trading. The current version includes a visualization tool for the Ichimoku Cloud system.

Predicting %change in price of USD/JPY currency pair with a LSTM. ForexBot 1.0 was an attempt to capture future price movement by shifting through ranges of previous periods. Issues arised with overfitting when increasing the range size of the period shifts being streamed into the algorithm. My implementation of Ichimoku caused future positions to leak through the indicators. Checkout NeoCandles to see an attempt to design a better method of  bundling the candle data. My next attempt will be to stream live image data of the Ichimoku into an CNN to find patterns in images.
