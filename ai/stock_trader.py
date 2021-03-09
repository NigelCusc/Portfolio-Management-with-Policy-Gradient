'''
Citation:
    Heavily inspired by this work:
    @Author: Louis Liang
    https://github.com/liangzp
    Several of his papers are discussed in my report
'''

import math
from decimal import Decimal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

eps = 10e-8
epochs = 0
M = 0

class StockTrader():

    def __init__(self):
        self.reset()
        self.eps = 0
        self.epochs = 0
        self.M = 0
        self.wealth = 10e3
        self.total_reward = 0
        self.ep_ave_max_q = 0
        self.loss = 0
        self.actor_loss = 0

        self.wealth_history = []
        self.r_history = []
        self.w_history = []
        self.p_history = []

    def reset(self):
        self.wealth = 10e3
        self.total_reward = 0
        self.ep_ave_max_q = 0
        self.loss = 0
        self.actor_loss = 0

        self.wealth_history = []
        self.r_history = []
        self.w_history = []
        self.p_history = []

    def update_summary(self, loss, r, q_value, actor_loss, w, p):
        self.loss += loss
        self.actor_loss += actor_loss
        self.total_reward += r
        self.ep_ave_max_q += q_value
        self.r_history.append(r)
        self.wealth = self.wealth * math.exp(r)
        self.wealth_history.append(self.wealth)
        self.w_history.extend([','.join([str(Decimal(str(w0)).quantize(Decimal('0.00'))) for w0 in w.tolist()[0]])])
        self.p_history.extend([','.join([str(Decimal(str(p0)).quantize(Decimal('0.000'))) for p0 in p.tolist()])])

    def write(self, codes, path_prefix):
        wealth_history = pd.Series(self.wealth_history)
        r_history = pd.Series(self.r_history)
        w_history = pd.Series(self.w_history)
        p_history = pd.Series(self.p_history)
        history = pd.concat([wealth_history, r_history, w_history, p_history], axis=1)
        history.to_csv(path_prefix + '-'.join(codes) + '-' + str(math.exp(np.sum(self.r_history)) * 100) + '.csv')

    def print_result(self, epoch, agent):
        self.total_reward = math.exp(self.total_reward) * 100
        print('*-----Episode: {:d}, Reward:{:.6f}%-----*'.format(epoch, self.total_reward))
        agent.write_summary(self.total_reward)
        agent.save_model()

    def plot_result(self):
        pd.Series(self.wealth_history).plot()
        plt.show()

    def action_processor(self, a, ratio):
        a = np.clip(a + ratio, 0, 1)
        a = a / (a.sum() + eps)
        return a
