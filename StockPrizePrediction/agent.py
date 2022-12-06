#!/usr/bin/python
# -*- coding: utf-8 -*-

from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
import random
from collections import deque


class Agent:
    def __init__(self, state_size, is_eval=False, model_name="", ):
        self.__inventory = []
        self.__total_profit = 0
        self.actions_history = []
        self.state_size = state_size
        self.action_size = 3 # sit, buy, sell
        self.memory = deque(maxlen=1000)
        self.model_name = model_name
        self.is_eval = is_eval
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        self.model = load_model("models/" + model_name) if is_eval else self.create_model()

    def create_model(self):
        # input size of window - output size of actions [3]
        model = Sequential()
        model.add(Dense(units=32, input_dim=2*self.state_size, activation="relu"))
        model.add(Dense(units=8, activation="relu"))
        model.add(Dense(units=self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(learning_rate=0.001))
        return model

    def reset(self):
        self.__inventory = []
        self.__total_profit = 0
        self.actions_history = []

    def action(self, state, price_data):
        if not self.is_eval and np.random.rand() <= self.epsilon:
            action = random.randrange(self.action_size)
        else:
            actions = self.model.predict(state)
            action = np.argmax(actions[0])
        bought_price = None
        if action == 0: # do nothing
            print(".", end='', flush=True)
            self.actions_history.append(action)
        elif action == 1: #buy
            self.buy(price_data)
            self.actions_history.append(action)
        elif action == 2 and self.has_inventory(): #sell
            bought_price = self.sell(price_data)
            self.actions_history.append(action)
        else:
            self.actions_history.append(0)
        return action, bought_price

    def buy(self, price_data):
        self.__inventory.append(price_data)
        print(f'Buy: {self.format_price(price_data)}')

    def sell(self, price_data):
        bought_price = self.__inventory.pop(0)
        profit = price_data - bought_price
        self.__total_profit += profit
        print(f'Sell: {self.format_price(price_data)} | Profit{self.format_price(profit)}')
        return bought_price

    def has_inventory(self):
        return len(self.__inventory) > 0

    def format_price(self, n):
        return ("-$" if n < 0 else "$") + f"{abs(n)}"

    def get_total_profit(self):
        return self.format_price(self.__total_profit)

    def experience_replay(self, batch_size):
        mini_batch = []
        l = len(self.memory)
        for i in range(l - batch_size + 1, l):
            mini_batch.append(self.memory[i])
        for state, action, reward, next_state, done in mini_batch:
            if done:
                target = reward
            else:
                next_q_values = self.model.predict(next_state)[0]
                target = reward + self.gamma * np.amax(next_q_values)

            predicted_target = self.model.predict(state)
            predicted_target[0][action] = target
            self.model.fit(state, predicted_target, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

