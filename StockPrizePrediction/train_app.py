#!/usr/bin/python
# -*- coding: utf-8 -*-

from agent import Agent
from market_env import Market

import os
import time


def main():
    # params
    windows_size = 5
    episode_count = 5
    stock_name = "GSPC_2011"
    batch_size = 32

    agent = Agent(windows_size)
    market = Market(windows_size, stock_name)

    start_time = time.time()
    for episode in range(episode_count + 1):
        print(f"Episode: {episode}")
        agent.reset()
        state, price_data = market.reset()

        for t in range(market.last_data_index):
            action, bought_price = agent.action(state, price_data)

            next_state, next_price_data, reward, done = market.get_next_state_and_reward(action, bought_price)
            agent.memory.append((state, action, reward, next_state, done))

            if len(agent.memory) > batch_size:
                agent.experience_replay(batch_size)

            state, price_data = next_state, next_price_data

            if done:
                print(f'{40* "-"}')
                print(f'Total profit: {agent.get_total_profit()}')
                print(f'{40 * "-"}')
        if episode % 10 == 0:
            if not os.path.exists("models"):
                os.mkdir("models")
            agent.model.save("models/model_episode" + str(episode))
    end_time = time.time()
    print(f"Computing training time: {end_time - start_time} seconds")


if __name__ == "__main__":
    main()