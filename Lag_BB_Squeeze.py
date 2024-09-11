import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import backtrader as bt
import numpy as np

# a. Extract daily data from Yahoo Finance for 10 years
ticker = "ITC.NS"
start_date = "2024-1-1"
end_date = "2024-2-19"
data = yf.download(ticker, start=start_date, end=end_date,interval='1h')

# b. Plot the chart for all data set
data['Close'].plot(figsize=(10, 6), title=f"{ticker} Stock Price")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.show()

# c. Backtest the modified strategy for multiple trades
class Strat2(bt.Strategy):
    params = (
        ('bb_period', 20),           # Period for Bollinger Bands calculation
        ('trade_units', 1),          # Number of units to trade
    )

    def __init__(self):
        self.first_trade = True
        self.bollinger_bands = bt.indicators.BollingerBands(self.data.close, period=self.params.bb_period)

    def next(self):
        bb_width = np.asarray(self.bollinger_bands.lines.top) - np.asarray(self.bollinger_bands.lines.bot)
    
        if not self.position:
            if self.first_trade and bb_width[0] < bt.indicators.StdDev(bb_width, period=self.params.bb_period)[0]:
                self.buy(size=self.params.trade_units)
                self.first_trade = False
            elif bb_width[0] < bt.indicators.StdDev(bb_width, period=self.params.bb_period)[0]:
                self.buy(size=self.params.trade_units)
        elif bb_width[0] > bt.indicators.StdDev(bb_width, period=self.params.bb_period)[0]:
            self.sell(size=self.params.trade_units)

    def start(self):
        # Set the original amount at the beginning of the backtest
        self.original_amount = 10 * self.data.close[0]
        self.broker.set_cash(self.original_amount)

    def stop(self):
        # Calculate percentage profit or loss
        final_value = self.broker.getvalue()
        percentage_change = ((final_value - self.original_amount) / self.original_amount) * 100

        print(f"\nOriginal Amount Invested: {self.original_amount:.2f}")
        print(f"Final Portfolio Value: {final_value:.2f}")
        print(f"Percentage Profit/Loss: {percentage_change:.2f}%")

# Create a backtest engine
cerebro = bt.Cerebro()
cerebro.adddata(bt.feeds.PandasData(dataname=data))
cerebro.addstrategy(Strat2)
cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')

# d. Print the results
results = cerebro.run()

cerebro.plot(style='candlestick', volume=True)

trade_analyzer = results[0].analyzers.trade_analyzer
print("\nTrade Analysis Results:")

# Check if 'total' and 'closed' attributes exist before accessing them
total_closed = trade_analyzer.get_analysis().get('total', {}).get('closed', 0)

if total_closed > 0:
    print(f"Total Trades: {total_closed}")
    print(f"Total Net Profit/Loss: {trade_analyzer.get_analysis().pnl.net.total:.2f}")

    # Check if 'trades' attribute exists before accessing it
    trades = trade_analyzer.get_analysis().get('trades', [])

    if trades:
        # Print individual trade details
        for i, trade in enumerate(trades):
            print(f"\nTrade {i + 1}:")
            entry_date = trade.getentry().datetime.strftime("%Y-%m-%d %H:%M:%S")
            exit_date = trade.getexit().datetime.strftime("%Y-%m-%d %H:%M:%S") if trade.isclosed else "N/A"

            print(f"  Entry Date: {entry_date}")
            print(f"  Entry Price: {trade.getentry().price:.2f}")
            print(f"  Exit Date: {exit_date}")
            print(f"  Exit Price: {trade.getexit().price:.2f}")
            print(f"  PnL: {trade.pnl:.2f}")
            print(f"  Status: {'Profit' if trade.pnl > 0 else 'Loss'}")
    else:
        print("No individual trade details available.")
else:
    print("No trades were executed.")
