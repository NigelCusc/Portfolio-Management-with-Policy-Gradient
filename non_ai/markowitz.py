import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from non_ai.generic_functions import *
import math


class Markowitz:
    def __init__(self):
        self.codes_df = pd.DataFrame()                      # Train DataFrame
        self.codes = []                                     # Portfolio Asset Codes
        self.sim_df = pd.DataFrame()                        # DataFrame of Simulations
        self.df_sharpe = pd.DataFrame()                     # Sharpe Ratio
        self.df_min_vol = pd.DataFrame()                    # Minimum Volatility
        self.df_max_ret = pd.DataFrame()                    # Maximum Return
        self.df_max_div = pd.DataFrame()                    # Maximum Diversity
        self.weighted_returns_sharpe = pd.DataFrame()       # Weighted Returns (TRAIN)
        self.weighted_returns_min_vol = pd.DataFrame()
        self.weighted_returns_max_ret = pd.DataFrame()
        self.weighted_returns_max_div = pd.DataFrame()
        self.port_ret_sharpe = pd.DataFrame()               # Portfolio Returns (TRAIN)
        self.port_ret_min_vol = pd.DataFrame()
        self.port_ret_max_ret = pd.DataFrame()
        self.port_ret_max_div = pd.DataFrame()
        self.port_ret_test_sharpe = pd.DataFrame()          # Portfolio Returns (TEST)
        self.port_ret_test_min_vol = pd.DataFrame()
        self.port_ret_test_max_ret = pd.DataFrame()
        self.port_ret_test_max_div = pd.DataFrame()
        self.cumulative_ret_sharpe = pd.DataFrame()         # Cumulative Returns (TRAIN)
        self.cumulative_ret_min_vol = pd.DataFrame()
        self.cumulative_ret_max_ret = pd.DataFrame()
        self.cumulative_ret_max_div = pd.DataFrame()
        self.cumulative_ret_test_sharpe = pd.DataFrame()    # Cumulative Returns (TEST)
        self.cumulative_ret_test_min_vol = pd.DataFrame()
        self.cumulative_ret_test_max_ret = pd.DataFrame()
        self.cumulative_ret_test_max_div = pd.DataFrame()
        self.returns_results_table = pd.DataFrame()         # Table of Results

    def Train(self, codes_df, codes, simulation_num=100000, balance=1):
        # Get Mean and Standard Deviation Moments for each Stock Item
        for asset in codes:
            # Get Mean
            codes_df[str(asset)]["Mean"] = mean(codes_df[str(asset)]["Log Return"])
            # Get Variance
            codes_df[str(asset)]["Std"] = stddev(codes_df[str(asset)]["Log Return"])

        self.codes_df = codes_df
        self.codes = codes

        # Simulation Phase =====================================
        print("Creating simulations...")
        # Set seed for reproducibility
        np.random.seed(0)
        # Randomize a Numpy Array with 2000 x n array
        rand_nos = np.random.rand(simulation_num, len(codes))
        # Create Randomly simulated weights
        simulated_portfolio_weights = rand_nos.transpose() / rand_nos.sum(axis=1)  # This is a 2000 x 1 array
        simulated_portfolio_weights = simulated_portfolio_weights.transpose()  # This is now a n x 2000 array
        # Put in a DataFrame
        df = pd.DataFrame(simulated_portfolio_weights, columns=codes)

        # Add Expected Return and Volatility in DataFrame =========
        df['Expected Return'] = PortfolioExpectedReturn(codes_df, simulated_portfolio_weights, codes)
        df['Volatility'] = PortfolioVolatility(codes_df, simulated_portfolio_weights, codes)
        df['Diversity Ratio'] = PortfolioDiversityRatio(codes_df, simulated_portfolio_weights, codes)
        self.sim_df = df

        # Locate Positions =========================================
        print("Locating portfolio positions...")
        # locate position of portfolio with lowest volatility
        min_vol_port = df.iloc[df['Volatility'].idxmin()]
        # locate position of portfolio with highest Sharpe Ratio
        max_sharpe = df.iloc[(df['Expected Return'] / df['Volatility']).idxmax()]
        # locate position of portfolio with highest Expected Return
        max_ret_port = df.iloc[df['Expected Return'].idxmax()]
        # locate position of portfolio with greatest diversification - (from medium link)
        max_div_port = df.iloc[df['Diversity Ratio'].idxmax()]

        # Get Weights For Positions Found ==========================
        print("Getting weights for portfolios located...")
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

        # Get Weighted Returns =====================================
        print("Calculating weighted returns...")
        self.weighted_returns_sharpe = pd.DataFrame()
        self.weighted_returns_min_vol = pd.DataFrame()
        self.weighted_returns_max_ret = pd.DataFrame()
        self.weighted_returns_max_div = pd.DataFrame()
        for asset in self.codes:
            self.weighted_returns_sharpe[str(asset)] = (
                    self.codes_df[str(asset)]['Daily Return'] * self.df_sharpe[str(asset)][0])
            self.weighted_returns_min_vol[str(asset)] = (
                    self.codes_df[str(asset)]['Daily Return'] * self.df_min_vol[str(asset)][0])
            self.weighted_returns_max_ret[str(asset)] = (
                    self.codes_df[str(asset)]['Daily Return'] * self.df_max_ret[str(asset)][0])
            self.weighted_returns_max_div[str(asset)] = (
                    self.codes_df[str(asset)]['Daily Return'] * self.df_max_div[str(asset)][0])

        # Sum of Weighted Returns ==================================
        print("Calculating sum of weighted returns...")
        self.port_ret_sharpe = self.weighted_returns_sharpe.sum(axis=1)  # axis = 1, means count rows
        self.port_ret_min_vol = self.weighted_returns_min_vol.sum(axis=1)
        self.port_ret_max_ret = self.weighted_returns_max_ret.sum(axis=1)
        self.port_ret_max_div = self.weighted_returns_max_div.sum(axis=1)

        # Cumulative Returns, Starting with a balance ===============
        print("Calculating cumulative returns...")
        self.cumulative_ret_sharpe = (balance - 1) + (self.port_ret_sharpe + 1).cumprod()
        self.cumulative_ret_min_vol = (balance - 1) + (self.port_ret_min_vol + 1).cumprod()
        self.cumulative_ret_max_ret = (balance - 1) + (self.port_ret_max_ret + 1).cumprod()
        self.cumulative_ret_max_div = (balance - 1) + (self.port_ret_max_div + 1).cumprod()

        print("Done!")

    def PlotSimulations(self):
        self.sim_df.plot(x='Volatility', y='Expected Return', style='o', title='Volatility vs. Expected Return')
        self.sim_df.plot(x='Volatility', y='Diversity Ratio', style='o', title='Volatility vs. Diversity Ratio')
        self.sim_df.plot(x='Diversity Ratio', y='Expected Return', style='o', title='Diversity Ratio vs. Expected Return')

    def PlotPortfolioPositions(self):
        df = self.sim_df
        # locate position of portfolio with lowest volatility
        min_vol_port = df.iloc[df['Volatility'].idxmin()]
        # locate position of portfolio with highest Sharpe Ratio
        max_sharpe = df.iloc[(df['Expected Return'] / df['Volatility']).idxmax()]
        # locate position of portfolio with highest Expected Return
        max_ret_port = df.iloc[df['Expected Return'].idxmax()]
        # locate position of portfolio with greatest diversification - (from medium link)
        max_div_port = df.iloc[df['Diversity Ratio'].idxmax()]

        # create scatter plot coloured by VaR
        plt.subplots(figsize=(15, 10))
        plt.scatter(df['Volatility'], df['Expected Return'], c=df['Volatility'], cmap='RdYlBu')
        plt.xlabel('Volatility')
        plt.ylabel('Expected Return')
        plt.title('Monte-carlo Simulation - Showing Markowitz Efficient Frontier')
        plt.colorbar()
        plt.margins(x=-0.45, y=-0.35)
        plt.xlim(0, 0.00015)

        # plot yellow star to highlight position of lowest variance portfolio
        plt.scatter(min_vol_port['Volatility'], min_vol_port['Expected Return'], marker=(5, 1, 0), color='y', s=500)
        # plot blue star to highlight position of highest Sharpe Ratio portfolio
        plt.scatter(max_sharpe['Volatility'], max_sharpe['Expected Return'], marker=(5, 1, 0), color='b', s=500)
        # plot green star to highlight position of highest return portfolio
        plt.scatter(max_ret_port['Volatility'], max_ret_port['Expected Return'], marker=(5, 1, 0), color='g', s=500)
        # plot red star to highlight position of highest portfolio diversity ratio
        plt.scatter(max_div_port['Volatility'], max_div_port['Expected Return'], marker=(5, 1, 0), color='r', s=500)

        plt.show()

    def Test(self, test_df, balance=1):
        print("Calculating returns...")
        self.port_ret_test_sharpe = GeneratePortfolioReturns(test_df, self.df_sharpe, self.codes)
        self.port_ret_test_min_vol = GeneratePortfolioReturns(test_df, self.df_min_vol, self.codes)
        self.port_ret_test_max_ret = GeneratePortfolioReturns(test_df, self.df_max_ret, self.codes)
        self.port_ret_test_max_div = GeneratePortfolioReturns(test_df, self.df_max_div, self.codes)

        print("Calculating cumulative returns...")
        self.cumulative_ret_test_sharpe = (balance - 1) + (self.port_ret_test_sharpe + 1).cumprod()
        self.cumulative_ret_test_min_vol = (balance - 1) + (self.port_ret_test_min_vol + 1).cumprod()
        self.cumulative_ret_test_max_ret = (balance - 1) + (self.port_ret_test_max_ret + 1).cumprod()
        self.cumulative_ret_test_max_div = (balance - 1) + (self.port_ret_test_max_div + 1).cumprod()
        print("Done!")

    def PlotCumulativeReturns(self, plot="Train"):
        # PLOT
        if plot == "Train":
            plt.figure(figsize=(8, 6), dpi=100)
            plt.plot(self.cumulative_ret_sharpe, label="Maximum Sharpe Ratio")
            plt.plot(self.cumulative_ret_min_vol, label="Minimum Volatility")
            plt.plot(self.cumulative_ret_max_ret, label="Maximum Expected Return")
            plt.plot(self.cumulative_ret_max_div, label="Maximum Diversity Ratio")

            plt.title('Portfolio Cumulative Returns (TRAIN)')
            plt.xlabel('Date')
            plt.ylabel('Cumulative Returns')
        else:
            plt.figure(figsize=(8, 6), dpi=100)
            plt.plot(self.cumulative_ret_test_sharpe, label="Maximum Sharpe Ratio")
            plt.plot(self.cumulative_ret_test_min_vol, label="Minimum Volatility")
            plt.plot(self.cumulative_ret_test_max_ret, label="Maximum Expected Return")
            plt.plot(self.cumulative_ret_test_max_div, label="Maximum Diversity Ratio")

            plt.title('Portfolio Cumulative Returns (TEST)')
            plt.xlabel('Date')
            plt.ylabel('Cumulative Returns')

        plt.legend()
        plt.show()

    def ReturnsResultsTable(self, plot="Train"):
        df = pd.DataFrame(columns=['Portfolio', 'Average Daily Yield', 'Sharpe Ratio', 'Maximum Drawdown'])
        if plot == "Train":
            df = self.ReturnsResultsTableRow(df, 'Maximum Sharpe Ratio', self.port_ret_sharpe)
            df = self.ReturnsResultsTableRow(df, 'Minimum Volatility', self.port_ret_min_vol)
            df = self.ReturnsResultsTableRow(df, 'Maximum Expected Return', self.port_ret_max_ret)
            df = self.ReturnsResultsTableRow(df, 'Maximum Diversity Ratio', self.port_ret_max_div)
        else:
            df = self.ReturnsResultsTableRow(df, 'Maximum Sharpe Ratio', self.port_ret_test_sharpe)
            df = self.ReturnsResultsTableRow(df, 'Minimum Volatility', self.port_ret_test_min_vol)
            df = self.ReturnsResultsTableRow(df, 'Maximum Expected Return', self.port_ret_test_max_ret)
            df = self.ReturnsResultsTableRow(df, 'Maximum Diversity Ratio', self.port_ret_test_max_div)
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



