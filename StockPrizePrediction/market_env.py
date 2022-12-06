#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn import preprocessing


class Market:
    def __init__(self, windows_size, stock_name):
        self.data = self.get_stock_data(stock_name)
        self.states = self.get_all_window_prices_diff(self.data, windows_size)
        self.index = -1
        self.last_data_index = len(self.data) - 1

    def get_stock_data(self, stock_name):
        # lines = open("data/" + stock_name + '.csv', 'r').read().splitlines()
        file_path = 'data/' + stock_name + ".csv"
        lines = pd.read_csv(file_path, sep=',')
        return lines

    def normalize_data(self, in_df):
        x = in_df.values
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df = pd.DataFrame(x_scaled, columns=in_df.columns)
        return df

    def get_all_window_prices_diff(self, data, windows_size):
        processed_data = []
        l = len(data)
        sel_col = ["Close", "Volume"]
        scaled_data = self.normalize_data(data[sel_col])
        for t in range(l):
            state = self.get_window(scaled_data, t, windows_size + 1)
            processed_data.append(state)
        return processed_data

    def get_window(self, data_df, t, n):
        d = t - n + 1
        data1 = data_df["Close"].values
        data2 = data_df["Volume"].values
        block1 = data1[d:t + 1] if d >= 0 else np.append(-d * [data1[0]], data1[0:t + 1])
        block2 = data2[d:t + 1] if d >= 0 else np.append(-d * [data2[0]], data2[0:t + 1])
        # block = data[d:t+1] if d >= 0 else -d * [data[0]] + data[0: t+1]
        res = []
        for i in range(n-1):
            res.append(block1[i+1] - block1[i])

        for i in range(n - 1):
            res.append(block2[i + 1] - block2[i])
        return np.array([res])

    def reset(self):
        self.index = -1
        return self.states[0], self.data.iloc[0]["Close"]

    def get_next_state_and_reward(self, action, bought_price=None):
        self.index += 1
        if self.index > self.last_data_index:
            self.index = 0
        next_state = self.states[self.index + 1]
        next_price_data = self.data.iloc[self.index + 1]["Close"]
        current_price_data = self.data.iloc[self.index]["Close"]
        reward = 0
        if action == 2 and bought_price is not None:   # 0 Holding, 1 Buy, 2 Selling
            reward = max(current_price_data - bought_price, 0)
        done = True if self.index == self.last_data_index - 1 else False

        return next_state, next_price_data, reward, done
