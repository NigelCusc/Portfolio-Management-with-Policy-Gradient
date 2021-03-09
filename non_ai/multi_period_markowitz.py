import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from non_ai.generic_functions import *
import math
from datetime import datetime, timedelta


class MultiPeriodMarkowitz:
    def __init__(self):
        self.simulation_num = 0
        self.balance = 0
        self.tc_fixed = 0  # Transaction Cost Fixed Price
        self.codes = []  # Portfolio Asset Codes
        self.actual_date_start = datetime.now()
        self.actual_date_end = datetime.now()
        self.window_size = 1
        self.rebalance_days = 1
        self.rebalance_days_arr = []    # Array of Rebalance dates to plot
        self.day = 1    # Current day. When this reaches the rebalance_days, the portfolio is rebalanced
        self.full_df = pd.DataFrame()

        # Weights
        self.df_sharpe = pd.DataFrame()  # Sharpe Ratio
        self.df_min_vol = pd.DataFrame()  # Minimum Volatility
        self.df_max_ret = pd.DataFrame()  # Maximum Return
        self.df_max_div = pd.DataFrame()  # Maximum Diversity
        # To check for weight change before and after train.
        # These will temporarily hold the weights before Training.
        self.df_sharpe_prev = pd.DataFrame()  # Sharpe Ratio
        self.df_min_vol_prev = pd.DataFrame()  # Minimum Volatility
        self.df_max_ret_prev = pd.DataFrame()  # Maximum Return
        self.df_max_div_prev = pd.DataFrame()  # Maximum Diversity

        self.weighted_returns_sharpe = dict()   # TEST Weighted returns
        self.weighted_returns_min_vol = dict()
        self.weighted_returns_max_ret = dict()
        self.weighted_returns_max_div = dict()

        self.test_ret_sharpe = pd.DataFrame()  # Returns (TEST)
        self.test_ret_min_vol = pd.DataFrame()
        self.test_ret_max_ret = pd.DataFrame()
        self.test_ret_max_div = pd.DataFrame()
        self.cumulative_ret_sharpe = pd.DataFrame()    # Cumulative Returns (TEST)
        self.cumulative_ret_min_vol = pd.DataFrame()
        self.cumulative_ret_max_ret = pd.DataFrame()
        self.cumulative_ret_max_div = pd.DataFrame()

        self.asset_balance_sharpe = dict()    # Initial Balance for separate assets
        self.asset_balance_min_vol = dict()
        self.asset_balance_max_ret = dict()
        self.asset_balance_max_div = dict()

        self.returns_results_table = pd.DataFrame()  # Result Table

    def Main(self, full_df, codes, date_start, date_end, window_size, rebalance_days, simulation_num=5000, balance=1,
             tc_fixed=0):
        # Get Mean and Standard Deviation Moments for each Stock Item
        for asset in codes:
            # Get Mean
            full_df[str(asset)]["Mean"] = mean(full_df[str(asset)]["Log Return"])
            # Get Variance
            full_df[str(asset)]["Std"] = stddev(full_df[str(asset)]["Log Return"])

        # Initialize variables
        self.full_df = full_df
        self.codes = codes
        self.window_size = window_size
        self.rebalance_days = rebalance_days
        self.rebalance_days_arr = []
        self.actual_date_start = date_start
        self.actual_date_end = date_end
        # Initially set the day as rebalance day, to cause initial weights
        self.day = rebalance_days
        self.simulation_num = simulation_num
        self.balance = balance
        self.tc_fixed = tc_fixed  # Transaction Cost Fixed Price

        # Custom Rolling Function
        self.LetsRoll(date_start, date_end)
        # Get Cumulative Returns
        self.GetCumulativeReturns()

        print("Done!")

    # Recursive Function to perform transactions
    def LetsRoll(self, date_start, date_end):
        # Check if end of series reached
        if date_start <= (date_end - timedelta(days=self.window_size)):
            # Get window to use as Train
            w_open = date_start
            w_close = w_open + timedelta(days=self.window_size)
            t = w_close + timedelta(days=1)     # The next day
            w_df = self.DictLoc(self.full_df, w_open, w_close)      # Train
            t_df = self.DictLoc(self.full_df, t, t)                 # Test
            self.Rebalancer(w_df, t_df, t)
            # Recursive
            self.LetsRoll((date_start + timedelta(days=1)), date_end)

    def DictLoc(self, full_df, start, end):
        ret_dict = dict()
        for asset in self.codes:
            ret_dict[str(asset)] = full_df[str(asset)].loc[start:end]
        return ret_dict

    def Rebalancer(self, df, t_df, t):   # Rebalance based on number of days
        # (Train df, test_df, test_day_string)
        # Get weights before:
        self.df_sharpe_prev = self.df_sharpe
        self.df_min_vol_prev = self.df_min_vol
        self.df_max_ret_prev = self.df_max_ret
        self.df_max_div_prev = self.df_max_div

        # Check if rebalance days reached
        if self.day >= self.rebalance_days:
            # Rebalance! (Train again)
            self.Train(df)
            self.day = 1

            # Add to Rebalance day string array
            self.rebalance_days_arr.append(t)
        else:
            # Use current weights
            self.day += 1
        # Test next day (Get weighted returns)
        self.Test(t_df)

    def Train(self, codes_df):
        # Simulation Phase =====================================
        # Set seed for reproducibility
        np.random.seed(0)
        # Randomize a Numpy Array with 2000 x n array
        rand_nos = np.random.rand(self.simulation_num, len(self.codes))
        # Create Randomly simulated weights
        simulated_portfolio_weights = rand_nos.transpose() / rand_nos.sum(axis=1)  # This is a 2000 x 1 array
        simulated_portfolio_weights = simulated_portfolio_weights.transpose()  # This is now a n x 2000 array
        # Put in a DataFrame
        df = pd.DataFrame(simulated_portfolio_weights, columns=self.codes)

        # Add Expected Return and Volatility in DataFrame =========
        df['Expected Return'] = PortfolioExpectedReturn(codes_df, simulated_portfolio_weights, self.codes)
        df['Volatility'] = PortfolioVolatility(codes_df, simulated_portfolio_weights, self.codes)
        df['Diversity Ratio'] = PortfolioDiversityRatio(codes_df, simulated_portfolio_weights, self.codes)

        # Locate Positions =========================================
        # locate position of portfolio with lowest volatility
        min_vol_port = df.iloc[df['Volatility'].idxmin()]
        # locate position of portfolio with highest Sharpe Ratio
        max_sharpe = df.iloc[(df['Expected Return'] / df['Volatility']).idxmax()]
        # locate position of portfolio with highest Expected Return
        max_ret_port = df.iloc[df['Expected Return'].idxmax()]
        # locate position of portfolio with greatest diversification - (from medium link)
        max_div_port = df.iloc[df['Diversity Ratio'].idxmax()]

        # Get Weights For Positions Found ==========================
        # Get weights used for highest sharpe ratio
        mask = (df['Expected Return'].values == max_sharpe['Expected Return']) & (
                df['Volatility'].values == max_sharpe['Volatility'])
        self.df_sharpe = df.loc[mask]
        self.df_sharpe.reset_index(inplace=True)

        # Get weights used for lowest variance
        mask1 = (df['Expected Return'].values == min_vol_port['Expected Return']) & (
                df['Volatility'].values == min_vol_port['Volatility'])
        self.df_min_vol = df.loc[mask1]
        self.df_min_vol.reset_index(inplace=True)

        # Get weights used for maximum expected return
        mask2 = (df['Expected Return'].values == max_ret_port['Expected Return']) & (
                df['Volatility'].values == max_ret_port['Volatility'])
        self.df_max_ret = df.loc[mask2]
        self.df_max_ret.reset_index(inplace=True)

        # Get weights used for maximum diversification
        mask3 = (df['Expected Return'].values == max_div_port['Expected Return']) & (
                df['Volatility'].values == max_div_port['Volatility'])
        self.df_max_div = df.loc[mask3]
        self.df_max_div.reset_index(inplace=True)

    def Test(self, test_df):
        # Get Weighted Returns =====================================
        for asset in self.codes:
            temp_weighted_returns_sharpe = (
                    test_df[str(asset)]['Daily Return'] * self.df_sharpe[str(asset)][0])
            temp_weighted_returns_min_vol = (
                    test_df[str(asset)]['Daily Return'] * self.df_min_vol[str(asset)][0])
            temp_weighted_returns_max_ret = (
                    test_df[str(asset)]['Daily Return'] * self.df_max_ret[str(asset)][0])
            temp_weighted_returns_max_div = (
                    test_df[str(asset)]['Daily Return'] * self.df_max_div[str(asset)][0])

            # Append these results (Weighted Returns) with the rest
            # MAX SHARPE
            if str(asset) in self.weighted_returns_sharpe:
                # Check if weight has changed
                if self.df_sharpe_prev[str(asset)][0] != self.df_sharpe[str(asset)][0]:
                    print("Asset: " + str(asset))
                    print("Weight Before: " + str(self.df_sharpe_prev[str(asset)][0]))
                    print("Weight Now: " + str(self.df_sharpe[str(asset)][0]))

                    print(temp_weighted_returns_sharpe)
                    # Get current Balance
                    #curr_asset_balance = (self.asset_balance_sharpe[str(asset)]) + \
                    #                     (self.weighted_returns_sharpe[str(asset)]).sum()
                    #print(curr_asset_balance)
                    # Reduce Percentage Transaction Cost using Current Balance
                    #temp_weighted_returns_sharpe -= (curr_asset_balance * (self.tc_perc / 100))
                    # Reduce Fixed Transaction Cost
                    diff = abs(self.df_sharpe_prev[str(asset)][0] - self.df_sharpe[str(asset)][0])
                    temp_weighted_returns_sharpe -= diff * self.tc_fixed
                    print(temp_weighted_returns_sharpe)
                    print("=========================================================================")
                self.weighted_returns_sharpe[str(asset)] = self.weighted_returns_sharpe[str(asset)]\
                    .append(temp_weighted_returns_sharpe)
            else:
                self.weighted_returns_sharpe[str(asset)] = temp_weighted_returns_sharpe
                # Take initial Balance based on weight
                print(str(asset) + " Initial Weight: " + str(self.df_sharpe[str(asset)][0]))
                self.asset_balance_sharpe[str(asset)] = self.df_sharpe[str(asset)][0] * self.balance

            # MIN VOL
            if str(asset) in self.weighted_returns_min_vol:
                # Check if weight has changed
                if self.df_min_vol_prev[str(asset)][0] != self.df_min_vol[str(asset)][0]:
                    # Get Current Balance
                    #curr_asset_balance = (self.asset_balance_min_vol[str(asset)]) + \
                     #                    (self.weighted_returns_min_vol[str(asset)]).sum()
                    # Reduce Percentage Transaction Cost using Current Balance
                    #temp_weighted_returns_min_vol -= curr_asset_balance * (self.tc_perc / 100)
                    # Reduce Fixed Transaction Cost
                    diff = abs(self.df_min_vol_prev[str(asset)][0] - self.df_min_vol[str(asset)][0])
                    temp_weighted_returns_min_vol -= diff * self.tc_fixed
                self.weighted_returns_min_vol[str(asset)] = self.weighted_returns_min_vol[str(asset)]\
                    .append(temp_weighted_returns_min_vol)
            else:
                self.weighted_returns_min_vol[str(asset)] = temp_weighted_returns_min_vol
                # Take initial Balance based on weight
                self.asset_balance_min_vol[str(asset)] = self.df_min_vol[str(asset)][0] * self.balance

            # MAX RET
            if str(asset) in self.weighted_returns_max_ret:
                # Check if weight has changed
                if self.df_max_ret_prev[str(asset)][0] != self.df_max_ret[str(asset)][0]:
                    # Get current Cumulative Return
                    #curr_asset_balance = (self.asset_balance_max_ret[str(asset)]) + \
                     #                    (self.weighted_returns_max_ret[str(asset)]).sum()
                    # Reduce Percentage Transaction Cost using Current Balance
                    #temp_weighted_returns_max_ret -= curr_asset_balance * (self.tc_perc / 100)
                    # Reduce Fixed Transaction Cost
                    diff = abs(self.df_max_ret_prev[str(asset)][0] - self.df_max_ret[str(asset)][0])
                    temp_weighted_returns_max_ret -= diff * self.tc_fixed
                self.weighted_returns_max_ret[str(asset)] = self.weighted_returns_max_ret[str(asset)]\
                    .append(temp_weighted_returns_max_ret)
            else:
                self.weighted_returns_max_ret[str(asset)] = temp_weighted_returns_max_ret
                # Take initial Balance based on weight
                self.asset_balance_max_ret[str(asset)] = self.df_max_ret[str(asset)][0] * self.balance

            # MAX DIV
            if str(asset) in self.weighted_returns_max_div:
                # Check if weight has changed
                if self.df_max_div_prev[str(asset)][0] != self.df_max_div[str(asset)][0]:
                    # Get current Cumulative Return
                    #curr_asset_balance = (self.asset_balance_max_div[str(asset)]) + \
                    #                     (self.weighted_returns_max_div[str(asset)]).sum()
                    # Reduce Percentage Transaction Cost using Cumulative Return
                    #temp_weighted_returns_max_div -= curr_asset_balance * (self.tc_perc / 100)
                    # Reduce Fixed Transaction Cost
                    diff = abs(self.df_max_ret_prev[str(asset)][0] - self.df_max_ret[str(asset)][0])
                    temp_weighted_returns_max_div -= diff * self.tc_fixed
                self.weighted_returns_max_div[str(asset)] = self.weighted_returns_max_div[str(asset)]\
                    .append(temp_weighted_returns_max_div)
            else:
                self.weighted_returns_max_div[str(asset)] = temp_weighted_returns_max_div
                # Take initial Balance based on weight
                self.asset_balance_max_div[str(asset)] = self.df_max_div[str(asset)][0] * self.balance

    def GetCumulativeReturns(self):
        # Sum of Weighted Returns ==================================
        print("Calculating sum of weighted returns...")
        # This time this is a dictionary, not a Dataframe
        weighted_returns_sharpe_df = pd.DataFrame.from_dict(self.weighted_returns_sharpe)
        weighted_returns_min_vol_df = pd.DataFrame.from_dict(self.weighted_returns_min_vol)
        weighted_returns_max_ret_df = pd.DataFrame.from_dict(self.weighted_returns_max_ret)
        weighted_returns_max_div_df = pd.DataFrame.from_dict(self.weighted_returns_max_div)

        self.test_ret_sharpe = weighted_returns_sharpe_df.sum(axis=1)  # axis = 1, means count rows
        self.test_ret_min_vol = weighted_returns_min_vol_df.sum(axis=1)
        self.test_ret_max_ret = weighted_returns_max_ret_df.sum(axis=1)
        self.test_ret_max_div = weighted_returns_max_div_df.sum(axis=1)

        # Cumulative Returns, Starting with a balance ===============
        print("Calculating cumulative returns...")
        self.cumulative_ret_sharpe = (self.balance - 1) + (self.test_ret_sharpe + 1).cumprod()
        self.cumulative_ret_min_vol = (self.balance - 1) + (self.test_ret_min_vol + 1).cumprod()
        self.cumulative_ret_max_ret = (self.balance - 1) + (self.test_ret_max_ret + 1).cumprod()
        self.cumulative_ret_max_div = (self.balance - 1) + (self.test_ret_max_div + 1).cumprod()

    def PlotCumulativeReturns(self):
        # PLOT
        plt.figure(figsize=(8, 6), dpi=100)
        plt.plot(self.cumulative_ret_sharpe, label="Maximum Sharpe Ratio")
        plt.plot(self.cumulative_ret_min_vol, label="Minimum Volatility")
        plt.plot(self.cumulative_ret_max_ret, label="Maximum Expected Return")
        plt.plot(self.cumulative_ret_max_div, label="Maximum Diversity Ratio")

        # Mark all Rebalance dates with vertical lines
        for xc in self.rebalance_days_arr:
            plt.axvline(x=xc, color='grey', linestyle='--')

        plt.title('Portfolio Cumulative Returns (Multi Period)')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Returns')

        plt.legend()
        plt.show()

    def ReturnsResultsTable(self):  # Only for TEST
        df = pd.DataFrame(columns=['Portfolio', 'Average Daily Yield', 'Sharpe Ratio', 'Maximum Drawdown'])
        df = self.ReturnsResultsTableRow(df, 'Maximum Sharpe Ratio', self.test_ret_sharpe)
        df = self.ReturnsResultsTableRow(df, 'Minimum Volatility', self.test_ret_min_vol)
        df = self.ReturnsResultsTableRow(df, 'Maximum Expected Return', self.test_ret_max_ret)
        df = self.ReturnsResultsTableRow(df, 'Maximum Diversity Ratio', self.test_ret_max_div)
        self.returns_results_table = df

    def ReturnsResultsTableRow(self, df, name, returns):
        df = df.append({'Portfolio': name,
                        'Average Daily Yield': round(float(np.mean(returns) * 100), 3),
                        'Sharpe Ratio': round(float(
                            np.mean(returns) / np.std(returns) * np.sqrt(252)), 3),
                        'Maximum Drawdown': round(float(max(
                            1 - min(returns) / np.maximum.accumulate(returns))), 3)
                        }, ignore_index=True)
        return df

