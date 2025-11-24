import pandas as pd
import numpy as np
import gym
import os
from gym import spaces
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
#from sb3_contrib import RecurrentPPO
from mplfinance.original_flavor import candlestick_ohlc

class TradingEnv(gym.Env):
    def _init_(self, csv_file, window_size=60, initial_balance=10000, lot_size=0.10, transaction_fee=0.001, max_position= 3):
        super(TradingEnv, self)._init_()
        
        # Load and preprocess data
        self.data = pd.read_csv(csv_file)
        self._validate_columns()  # New validation method
        self.prices = self.data[['open', 'high', 'low', 'close']].values
        self.max_position = max_position
        self.long_position = 0
        self.short_position = 0
        self.last_long_profit = 0
        self.last_short_profit = 0
        self.long_position_closed = False
        self.short_position_closed = False
        self.long_entry = None
        self.short_entry = None
        self.features = self._create_features()
        
        # Environment parameters
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.lot_size = lot_size
        self.transaction_fee = transaction_fee
        self.max_steps = len(self.data) - window_size - 1
        
        # Spaces
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=-5.0, 
            high=5.0,
            shape=(window_size, self.features.shape[1] + 5),
            dtype=np.float32
        )
        
        self.reset()

    def _validate_columns(self):
        required = ['PWAP', 'MACD', 'RSI', 'EMA12', 'open', 'high', 'low', 'close']
        missing = [col for col in required if col not in self.data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _create_features(self):
        # Use pre-calculated indicators from CSV
        features = self.data[['PWAP', 'MACD', 'RSI', 'EMA12']].values
        
        # Normalization
        means = np.nanmean(features, axis=0)
        stds = np.nanstd(features, axis=0)
        features = (features - means) / (stds + 1e-8)
        return np.nan_to_num(features)

    def reset(self):
        self.balance = self.initial_balance
        #self.position = 0  # Number of lots held
        self.long_position = 0
        self.short_position = 0
        self.long_entry = None
        self.short_entry = None
        self.entry_price = None
        self.current_step = self.window_size
        self.trades = []
        self.portfolio_history = [self.initial_balance] * self.window_size
        return self._next_observation()

    def _next_observation(self):
        # Market features
        obs = self.features[self.current_step-self.window_size:self.current_step]
        
        # Portfolio features
        portfolio_features = np.array([
            [self.balance / self.initial_balance,
             self.long_position / self.max_position,
             self.short_position / self.max_position,
             (self.long_position + self.short_position) * self.lot_size / 1000,
            #  self.long_entry / self.current_price if self.long_entry else 0,
            #  self.short_entry / self.current_price if self.short_entry else 0,
             self._current_value() / self.initial_balance]
            for _ in range(self.window_size)
        ])
        
        return np.concatenate([obs, portfolio_features], axis=1).astype(np.float32)

    def _current_value(self):
        long_value = self.long_position * self.lot_size * self.current_price if self.long_position > 0 else 0
    
    # Calculate short position value (margin + potential profit)
        short_value = 0
        if self.short_position > 0 and self.short_entry is not None:
            short_profit = (self.short_entry - self.current_price) * self.lot_size * self.short_position
            short_value = (self.short_position * self.lot_size * self.short_entry) + short_profit

        return self.balance + long_value + short_value ###self.position * self.lot_size * self.current_price

    @property
    def current_price(self):
        return self.prices[self.current_step, 3]  # Close price

    def step(self, action):
        self.current_step += 1
        prev_value = self._current_value()
        self.long_position_closed = False
        self.short_position_closed = False

        #33 for tarcking the hold
        if self.long_position > 0 or self.short_position > 0:
            self.holding_duration += 1
        else:
            self.holding_duration = 0
        
        # Execute action
        if action == 1:
            if self.short_position > 0:
                self._close_short_position()
            self._execute_buy()   
        elif action == 2:
            if self.long_position > 0:
                self._close_long_position() 
            self._execute_sell()
        elif action == 3:  # Close action
            #print(f"Closed {self.position} lots at step {self.current_step}")
            self._close_long_position()
        elif action == 4:
            self._close_short_position()


        new_value = self._current_value()
        reward = self._calculate_reward(prev_value, new_value)
        done = self.current_step >= len(self.data) - 1 or self.balance < 0
        
        self.portfolio_history.append(new_value)
        
        return self._next_observation(), reward, done, self._get_info()

    def _execute_buy(self):
        if self.long_position >= self.max_position:
            return    #### position limit also add tracking
        
        cost = self.current_price * self.lot_size * (1 + self.transaction_fee)
        if self.balance >= cost:
            self.balance -= cost
            self.long_position += 1
            #### set entry price
            if self.long_entry is None:
                self.long_entry = self.current_price
            # Add current step as 4th element
            self.trades.append(('buy', self.current_price, self.lot_size, self.current_step))

    def _execute_sell(self):
        # Short position execution
        if self.short_position >= self.max_position:
            return
            
        margin_required = self.current_price * self.lot_size * (1 + self.transaction_fee)
        if self.balance >= margin_required:
            self.balance -= margin_required
            self.short_position += 1
            if self.short_entry is None:
                self.short_entry = self.current_price
            self.trades.append(('sell', self.current_price, self.lot_size, self.current_step))
            #print(f"opened short at {self.current_price} at step {self.current_step}")

    def _close_long_position(self):
        if self.long_position > 0 and self.long_entry is not None:
            close_price = self.current_price
            #entry_price = self.long_entry
            ##  self.realized_long_pnl = (self.current_price - self.long_entry) * self.lot_size * self.long_position
            self.last_long_profit = (close_price - self.long_entry) / self.long_entry
            self.long_position_closed = True
            proceeds = self.long_position * close_price * self.lot_size * (1 - self.transaction_fee)
            self.balance += proceeds
            self.trades.append(('close_long', close_price, self.long_position * self.lot_size, self.current_step))
            self.long_position = 0
            self.long_entry = None
            

        # Close SHORT positions
    def _close_short_position(self):
        if self.short_position > 0 and self.short_entry is not None:
            close = self.current_price
            self.last_short_profit = (self.short_entry - close) * self.short_entry
            self.short_position_closed = True
            margin_return = self.short_position *self.lot_size * self.short_entry
            profit = (self.short_entry - close) * self.short_position * self.lot_size
            self.balance += margin_return + profit
            self.trades.append(('close_short', close, self.short_position, self.current_step))
            self.short_position = 0
            self.short_entry = None
            ###3 reste the entries & and track the positions
            

    def _calculate_reward(self, prev_value, new_value):
        """Reward based on closed positions only"""
        reward = 0
        
        # Long close reward
        if self.long_position_closed: #and self.long_entry is not None:
            #profit = (self.current_price - self.long_entry) / self.long_entry
            reward += self.last_long_profit * 2  # Amplify good long closes
            self.long_position_closed = False
            
        # Short close reward
        if self.short_position_closed and self.short_entry is not None:
            #profit = (self.short_entry - self.current_price) / self.short_entry
            reward += self.last_short_profit * 2  # Amplify good short closes\
            self.short_position_closed = False
            
        # Small penalty for holding positions
        if self.long_position > 0 or self.short_position > 0:
            reward -= 0.001 * self.holding_duration
            
        return reward

    def _get_info(self):
        return {
            'balance': self.balance,
            'long_position': self.long_position,
            'short_position': self.short_position,
            'value': self._current_value(),
            'price': self.current_price
        }

    def render(self, mode='human'):
        plt.figure(figsize=(15, 10))
        
        # Candlestick chart
        price_ax = plt.subplot(2,1,1)
        plotted_prices = self.prices[:self.current_step+1]
        ohlc = [(i, row[0], row[1], row[2], row[3]) 
               for i, row in enumerate(plotted_prices)] ## was self.prices
        candlestick_ohlc(plt.gca(), ohlc, width=0.6, colorup='g', colordown='r')
        
        if self.long_position > 0 or self.short_position > 0:
            start_step = self.current_step - self.holding_duration
            end_step = self.current_step
            price_ax.axvspan(start_step, end_step, color='yellow', alpha=0.3, label='Holding Period')

        ### for longing positions 
        if self.long_entry is not None:
            price_ax.axhline(self.long_entry, color='orange', linestyle='-', label='Current Price')
        if self.short_entry is not None:
            price_ax.axhline(self.short_entry, color='purple', linestyle='--', label='Short Entry Price')
        # Plot trades
        buy_dates = [t[3] for t in self.trades if t[0] == 'buy']
        sell_dates = [t[3] for t in self.trades if t[0] == 'sell']
        close_long_dates = [t[3] for t in self.trades if t[0] == 'close_long']
        close_short_dates = [t[3] for t in self.trades if t[0] == 'close_short']
        close_dates = [t[3] for t in self.trades if t[0] == 'close']
        
        price_ax.scatter(buy_dates, self.prices[buy_dates,3], marker='^', color='lime', 
                    s=30, label='Long Open', zorder=5)
        price_ax.scatter(sell_dates, self.prices[sell_dates,3], marker='v', color='magenta',
                        s=30, label='Short Open', zorder=5)
        price_ax.scatter(close_long_dates, self.prices[close_long_dates,3], marker='X',
                        color='darkgreen', s=20, label='Long Close', zorder=5)
        price_ax.scatter(close_short_dates, self.prices[close_short_dates,3], marker='X',
                        color='darkred', s=20, label='Short Close', zorder=5)
        price_ax.set_title('Price Action with Trades')
        price_ax.set_ylabel('Price')
        price_ax.legend()

        # Portfolio value
        plt.subplot(2,1,2)
        plt.plot(self.portfolio_history, label='Portfolio Value')
        plt.xlabel('Steps')
        plt.ylabel('Value') 
        plt.legend()
        
        plt.tight_layout()
        plt.show()

if _name_ == "_main_":
    env = TradingEnv('/kaggle/input/trainingg/RL2.csv', lot_size=0.01)
    logs = os.path.join('Training', 'logs')
    #model = RecurrentPPO('MlpLstmPolicy', env, verbose=1, tensorboard_log=logs, n_steps=4096)
    model = PPO('MlpPolicy', env, learning_rate=0.00065, batch_size=128, verbose=1, tensorboard_log=logs,device='cuda',n_steps=16384)
    model.learn(total_timesteps=6000000)
    model.save('waed')
