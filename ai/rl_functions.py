# Generic Functions utilized for Reinforcement Learning

import numpy as np
from ai.stock_trader import StockTrader
import math
import matplotlib.pyplot as plt
import pandas as pd


# Parse info provided from the Environment
def parse_info(info):
    return info['reward'], info['continue'], info['next state'], info['weight vector'], info['price'], info['risk']

# Performs the Traversal / Markov Decision Process
def traversal(stocktrader, agent, env, epoch, trainable):
    # STEP and get info on next step
    info = env.step(None, None)
    # reward, continue, next state, weight vector, price, risk
    r, contin, s, w1, p, risk = parse_info(info)

    # INITIALIZE
    contin = 1
    t = 0

    while contin:
        # Get updated weights from Agent
        w2 = agent.predict(s, w1)   # Given State and current weights

        # STEP
        env_info = env.step(w1, w2)
        r, contin, s_next, w1, p, risk = parse_info(env_info)

        # SAVE
        agent.save_transition(s, p, w2, w1)

        # INIT
        loss, q_value, actor_loss = 0, 0, 0

        if not contin and trainable == "True":
            agent.train()

        stocktrader.update_summary(loss, r, q_value, actor_loss, w2, p)
        s = s_next
        t = t + 1

# Calculate Maximum Drawdown
def maxdrawdown(arr):
    i = np.argmax((np.maximum.accumulate(arr) - arr) / np.maximum.accumulate(arr))  # end of the period
    j = np.argmax(arr[:i])  # start of period
    return (1 - arr[i] / arr[j])

# Test and Save Weights
def backtest(agent, env, PATH_prefix, initial_investment):
    # Initialize stocktrader
    stocktrader = StockTrader()

    # Initial Step
    info = env.step(None, None)
    r, contin, s, w1, p, risk = parse_info(info)

    # Initial investment
    contin = 1
    wealth = initial_investment
    cumulative_returns = [wealth]
    returns = []

    while contin:
        # Agent action (portfolio weights prediction) given current weights and state
        w2 = agent.predict(s, w1)
        # Pass action to environment and get next date
        env_info = env.step(w1, w2)
        # Get return and other environment info (next state, etc.)
        r, contin, s_next, w1, p, risk = parse_info(env_info)

        # Calculate wealth and add it to List
        wealth = wealth * math.exp(r)

        returns.append(math.exp(r) - 1)
        cumulative_returns.append(wealth)

        # Set state to next state from environment
        s = s_next
        stocktrader.update_summary(0, r, 0, 0, w2, p)

    # Create csv with weights
    stocktrader.write(map(lambda x: str(x), env.get_codes()), PATH_prefix)

    return returns, cumulative_returns


def PlotCumulativeReturns(cumulative_returns):
    # PLOT
    plt.figure(figsize=(8, 6), dpi=100)
    plt.plot(cumulative_returns, label="Policy Gradient")
    plt.title('Portfolio Cumulative Returns (Deep Reinforcement Learning)')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.show()

# Provide results to summarize the returns
def ReturnsResultsTable(returns):
    # RESULT
    df = pd.DataFrame(columns=['Portfolio', 'Average Daily Yield', 'Sharpe Ratio', 'Maximum Drawdown'])
    df = ReturnsResultsTableRow(df, 'Policy Gradient', returns)
    return df

# Used in the above method
def ReturnsResultsTableRow(df, name, returns):
    df = df.append({'Portfolio': name,
                    'Average Daily Yield': round(float(np.mean(returns) * 100), 3),
                    'Sharpe Ratio': round(float(
                        np.mean(returns) / np.std(returns) * np.sqrt(252)), 3),
                    'Maximum Drawdown': round(float(max(
                        1 - min(returns) / np.maximum.accumulate(returns))), 3)
                    }, ignore_index=True)
    return df
