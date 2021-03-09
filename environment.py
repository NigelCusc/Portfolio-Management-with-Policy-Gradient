# Environment shared throughout our experiments

import numpy as np
import pandas as pd
from math import log
from datetime import datetime
import time
import random

eps = 10e-8


def fill_zeros(x):
    return '0'*(6-len(x))+x


class Environment:
    def __init__(self):
        self.cost = 0.0025  # Transaction Cost
        self.codes = []             # Array of Assets/Codes
        self.data = pd.DataFrame()  # DataFrame that holds stock data
        self.date_set = []          # Full list of dates
        # Initial input shape
        self.M = 0      # Asset/code list length
        self.N = 0      # Feature list length
        self.L = 0      # Window length
        self.states = []
        self.price_history = []
        self.t = self.L + 1
        self.first_step_done_flag = False

    def set_transaction_cost(self, cost):
        self.cost = cost

    # Read csv and fill data field
    def read_csv(self, start_time, end_time, features):
        self.data = pd.read_csv(r'./data/stock_data.csv', index_col=0, parse_dates=True, dtype=object)
        self.data["Code"] = self.data["Code"].astype(str)
        self.data[features] = self.data[features].astype(float)
        self.data = self.data[start_time.strftime("%Y-%m-%d"):end_time.strftime("%Y-%m-%d")]

    # Return train_start_date, train_end_date, test_start_date, test_end_date, codes array
    def get_repo(self, start_date, end_date, codes_num=10):
        #preprocess parameters
        # read CSV
        self.data = pd.read_csv(r'./data/stock_data.csv', index_col=0, parse_dates=True, dtype=object)
        self.data["Code"] = self.data["Code"].astype(str)

        # Get a random sample of stock codes
        codes = random.sample(set(self.data["Code"]), codes_num)
        # Get a subset of the data
        data2 = self.data.loc[self.data["Code"].isin(codes)]

        # Get dates
        date_set = set(data2.loc[data2['Code'] == codes[0]].index)
        for code in codes:
            date_set = date_set.intersection((set(data2.loc[data2['Code'] == code].index)))

        date_set = date_set.intersection(set(pd.date_range(start_date, end_date)))
        self.date_set = list(date_set)
        self.date_set.sort()

        # Train Test Ratio
        train_start_time = self.date_set[0]
        train_end_time = self.date_set[int(len(self.date_set) / 3) * 2 - 1]
        test_start_time = self.date_set[int(len(self.date_set) / 3) * 2]
        test_end_time = self.date_set[-1]

        return train_start_time, train_end_time, test_start_time, test_end_time, codes

    # Set up environment for the selected timeseries
    def get_data(self, start_time, end_time, codes, features=['Adj Close'], window_length=3):
        self.codes = codes
        self.read_csv(start_time, end_time, features)

        # Initialize parameters
        self.M = len(codes)+1
        self.N = len(features)
        self.L = int(window_length)
        asset_dict = self.get_asset_dict(start_time, end_time, codes)
        self.states = []
        self.price_history = []

        # Initially we set the time as the Window Length
        t = self.L+1

        #self.date_set = pd.date_range(start_time, end_time)
        self.date_set = set(self.data.loc[self.data['Code'] == codes[0]].index)

        # Set up States and Price History
        length = len(self.date_set)
        print("Date set count: " + str(length))
        while t < length-1:
            # Array of 1s' with the size of the window length
            V_close = np.ones(self.L)

            y = np.ones(1)

            state = []  # Initialize state
            for asset in codes:
                asset_data = asset_dict[str(asset)]
                # stack asset window
                V_close = np.vstack((V_close, asset_data.loc[asset_data.index[t - self.L - 1:t - 1], 'Adj Close']))
                y = np.vstack((y, asset_data.loc[asset_data.index[t], 'Adj Close']/asset_data.loc[asset_data.index[t-1],
                                                                                                  'Adj Close']))
            state.append(V_close)

            state = np.stack(state, axis=1)
            state = state.reshape(1, self.M, self.L, self.N)

            self.states.append(state)
            self.price_history.append(y)

            t = t+1
        self.reset()

    # Returns dictionary with all asset data
    def get_asset_dict(self, start_time, end_time, codes, features=['Adj Close']):
        self.codes = codes
        self.read_csv(start_time, end_time, features)

        # Initialize parameters
        #self.date_set = pd.date_range(start_time, end_time)
        self.date_set = set(self.data.loc[self.data['Code'] == codes[0]].index)

        # Loop for each stock Code to fill dictionary
        asset_dict = dict()
        for asset in codes:
            asset_data = self.data[self.data["Code"] == asset].reindex(self.date_set).sort_index()

            # Fill missing dates with mean
            asset_data = asset_data.resample('D').mean()

            asset_data = asset_data.fillna(method='bfill', axis=1)
            asset_data = asset_data.fillna(method='ffill', axis=1)
            #***********************open as preclose*******************#
            asset_data = asset_data.dropna(axis=0, how='any')
            asset_dict[str(asset)] = asset_data

        return asset_dict

    # Step function for Training. Weights as input
    def step(self, w1, w2):
        # Check if first step
        if self.first_step_done_flag:
            not_terminal = 1
            # Get price at beginning of window
            price = self.price_history[self.t]

            # Effect Transaction Cost: (Cost * Number of Transactions)
            mu = self.cost * (np.abs(w2[0][0:] - w1[0][0:])).sum()


            # For now risk is set to 0. Future work could include using a risk measure to train.
            # Example Value at risk?
            risk = 0

            # Calculate return and reward
            r = (np.dot(w2, price)[0] - mu)[0]
            reward = np.log(r + eps)

            # Update weights
            w2 = w2 / (np.dot(w2, price) + eps)

            self.t += 1

            # Check if last step
            if self.t == len(self.states):
                not_terminal = 0
                self.reset()

            # Get rid of useless dimension arrays
            price = np.squeeze(price)

            # Set and return step info
            info = {
                        'reward': reward,
                        'continue': not_terminal,
                        'next state': self.states[self.t],
                        'weight vector': w2,
                        'price': price,
                        'risk': risk
                    }
            return info
        else:
            # Return first step
            info = {
                        'reward': 0,
                        'continue': 1,
                        'next state': self.states[self.t],
                        'weight vector': np.array([[1] + [0 for i in range(self.M-1)]]),    # array([[1, 0, 0, ...]])
                        'price': self.price_history[self.t],
                        'risk': 0
                    }

            self.first_step_done_flag = True
            return info

    # Reset environment
    def reset(self):
        # set time as the window length (to restart)
        self.t = self.L+1
        # Mark as first step
        self.first_step_done_flag = False

    # Return array of assets/codes
    def get_codes(self):
        return self.codes
