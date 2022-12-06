#!/usr/bin/python
# -*- coding: utf-8 -*-

from tensorflow.keras.models import load_model
from agent import Agent
from market_env import Market

import matplotlib.pyplot as plt


def main():
    windows_size = 5
    stock_name = "GSPC_2011-03"
    model_name = "model_episode10"
    model = load_model(f"models/" + model_name)
    # windows_size = model.layers[0].input.shape.as_list()[1]

    agent = Agent(windows_size, True, model_name)
    market = Market(windows_size, stock_name)
    state, price_data, = market.reset()

    for t in range(market.last_data_index):
        action, bought_price = agent.action(state, price_data)

        next_state, next_price_data, reward, done = market.get_next_state_and_reward(action, bought_price)
        state = next_state
        price_data = next_price_data

        if done:
            print(f'{40 * "-"}')
            print(f'{stock_name} | Total profit: {agent.get_total_profit()}')
            print(f'{40 * "-"}')

    plot_action_profit(market.data["Close"].values, agent.actions_history, agent.get_total_profit())


def plot_action_profit(data, action_data, profit):
    # plt.figure()
    plt.plot(range(len(data)), data)
    plt.xlabel("data")
    plt.ylabel("price")

    buy, sell = False, False
    for d in range(len(data) - 1):
        if action_data[d] == 1: # buy
            buy, = plt.plot(d, data[d], 'g*')
        elif action_data[d] == 2:
            sell, = plt.plot(d, data[d], 'r+')

    plt.legend(["Price", "Buy", "Sell"])
    plt.title(f"Total Profit: {profit}")
    plt.savefig("buy_sell.png")
    plt.show()


if __name__ == "__main__":
    main()