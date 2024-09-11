import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import backtrader as bt
import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define Q-learning strategy
class QLearningStrategy(bt.Strategy):
    params = dict(
        alpha=0.1,
        gamma=0.99,
        rsi_period=14,
        macd1=12,
        macd2=26,
        macdsig=9,
        vol_period=20,
        portfolio_norm=1e6,
        vol_bins=[-1.0, -0.5, 0.0, 0.5, 1.0],
        rsi_bins=[0, 30, 50, 70, 100],
        macd_bins=[-2.0, -1.0, 0.0, 1.0, 2.0],
        epsilon=1e-6  # Small epsilon to avoid division by zero
    )

    def __init__(self):
        self.rsi = bt.indicators.RSI(period=self.params.rsi_period)
        self.macd = bt.indicators.MACD(
            period_me1=self.params.macd1, period_me2=self.params.macd2, period_signal=self.params.macdsig)
        self.vol_ma = bt.indicators.SMA(self.data.volume, period=self.params.vol_period)
        self.normalized_vol = (self.data.volume - self.vol_ma) / self.vol_ma

        num_actions = 5  # [Hold, Buy, Sell, Buy Intraday, Sell Intraday]
        self.q_table = defaultdict(lambda: np.zeros(num_actions))
        self.prev_state = None
        self.prev_action = None
        self.prev_portfolio_value = None
        self.starting_cash = None

    def get_state(self):
        vol_bin = np.digitize(self.normalized_vol[0], self.params.vol_bins) - 1
        rsi_bin = np.digitize(self.rsi[0], self.params.rsi_bins) - 1
        macd_bin = np.digitize(self.macd.macd[0], self.params.macd_bins) - 1
        return (vol_bin, rsi_bin, macd_bin)

    def next(self):
        if self.starting_cash is None:
            self.starting_cash = self.data.close[0] * 100  # Initial cash is 100 times the first closing price
            self.broker.set_cash(self.starting_cash)
            self.prev_portfolio_value = self.broker.getvalue()

        state = self.get_state()
        r_a = self.calculate_r_a()

        # Determine the number of actions to consider based on r_a
        num_available_actions = max(1, int(r_a * len(self.q_table[state])))

        # Sort actions by their Q-values
        sorted_actions = np.argsort(self.q_table[state])[::-1]
        chosen_actions = sorted_actions[:num_available_actions]

        # Avoid division by zero and invalid probabilities
        q_values = self.q_table[state][chosen_actions]
        q_values = np.maximum(q_values, 0)  # Ensure Q-values are non-negative
        total_q = np.sum(q_values)
        if total_q == 0:
            probabilities = np.ones(len(q_values)) / len(q_values)  # Uniform distribution
        else:
            probabilities = q_values / total_q

        # Roulette wheel selection
        action = np.random.choice(chosen_actions, p=probabilities)

        if self.prev_state is not None:
            reward = self.calculate_reward()
            self.q_table[self.prev_state][self.prev_action] += self.params.alpha * (
                reward + self.params.gamma * np.max(self.q_table[state]) - self.q_table[self.prev_state][self.prev_action])

        self.prev_state = state
        self.prev_action = action

        # Execute action based on selection
        if action == 1 and self.broker.get_cash() > self.data.close[0]:
            self.buy()  # Buy
        elif action == 2 and self.position:  # Sell only if holding units
            self.sell()
        elif action == 3 and self.broker.get_cash() > self.data.close[0]:
            self.buy()  # Buy Intraday
        elif action == 4 and self.position:  # Sell Intraday only if holding units
            self.sell()
        # No need to do anything if action == 0 (Hold)

        # Update previous portfolio value after the action
        self.prev_portfolio_value = self.broker.getvalue()

    def calculate_r_a(self):
        # Calculate volatility as the standard deviation of recent price changes
        volatility = np.std([self.data.close[i] for i in range(-10, 0)])
        
        # Avoid division by zero
        portfolio_value = self.broker.getvalue()
        portfolio_change = abs((portfolio_value - self.starting_cash) / self.starting_cash) + self.params.epsilon
        x = volatility / portfolio_change  # Volatility divided by absolute profit or loss percent
        return sigmoid(x)  # Sigmoid transformation

    def calculate_reward(self):
        # Reward is the change in portfolio value between the previous and current time step
        current_portfolio_value = self.broker.getvalue()
        reward = current_portfolio_value - self.prev_portfolio_value
        return reward

    def notify_trade(self, trade):
        if trade.isclosed:
            # Force close any open positions by the end of the day
            if trade.size > 0 and self.position:
                self.sell()  # Force close

# Calculate the date one year ago from today
end_date = datetime.now()
start_date = end_date - timedelta(days=365)

# Download minute-wise data from one year ago to today
data = yf.download("CUMMINSIND.NS", start=start_date, end=end_date, interval="1h")

# Convert the data to Backtrader format
data_feed = bt.feeds.PandasData(dataname=data)

# Initialize Cerebro engine
cerebro = bt.Cerebro()
cerebro.addstrategy(QLearningStrategy)
cerebro.adddata(data_feed)
cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')

# Print initial portfolio value before running the strategy
print(f"Initial Portfolio Value: {cerebro.broker.startingcash}")

# Run the strategy
results = cerebro.run()

# Print final portfolio value after running the strategy
print(f"Final Portfolio Value: {cerebro.broker.getvalue()}")

# Plot results using Backtrader's built-in plotting
cerebro.plot(style='candlestick', volume=True)

# Extract trade analysis results
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
            entry_date = trade.getentry().datetime.strftime("%Y-%m-%d %H:%M:%S")
            exit_date = trade.getexit().datetime.strftime("%Y-%m-%d %H:%M:%S") if trade.isclosed else "N/A"

            print(f"\nTrade {i + 1}:")
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

# Optional: Plot additional trade analysis graphs if needed
def plot_trade_analysis(trades):
    if trades:
        entry_prices = [trade.getentry().price for trade in trades]
        exit_prices = [trade.getexit().price if trade.isclosed else None for trade in trades]
        pnl = [trade.pnl for trade in trades]

        plt.figure(figsize=(12, 6))
        plt.plot(entry_prices, label='Entry Prices', marker='o', linestyle='None')
        plt.plot(exit_prices, label='Exit Prices', marker='x', linestyle='None')
        plt.title('Trade Entry and Exit Prices')
        plt.xlabel('Trade Number')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(12, 6))
        plt.plot(pnl, label='Profit/Loss', marker='o', linestyle='-')
        plt.title('Trade Profit/Loss')
        plt.xlabel('Trade Number')
        plt.ylabel('PnL')
        plt.legend()
        plt.grid(True)
        plt.show()

# Plot trade analysis
plot_trade_analysis(trades)
