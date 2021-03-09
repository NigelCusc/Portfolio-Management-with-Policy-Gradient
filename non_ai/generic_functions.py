import numpy as np
import pandas as pd
import math

def mean(data):
    """Return the sample arithmetic mean of data."""
    # Get count/length of dataset
    n = len(data)
    # Checking if data is empty
    if n < 1:
        raise ValueError('mean requires at least one data point')

    # Divide sum of all data with the count of dataset
    return sum(data) / n


def median(data):
    """Return the middle point of the ordered list."""
    # Get count/length of dataset
    n = len(data)

    # Order the dataset
    s = sorted(data)

    # Find middle point
    return (sum(s[n // 2 - 1:n // 2 + 1]) / 2.0, s[n // 2])[n % 2] if n else None


def _ss(data, n):
    """
    Return sum of square/cubic/quadratic deviations of sequence data.
    data: Dataset
    n: number of derivations. e.g. 2-square, 3-cubic, 4-quadratic
    """
    # Get mean of our dataset
    c = mean(data)
    '''
    foreach item in the dataset, get its difference to the mean (c) - [Hence the deviation]
    and square the result by the number of derivations (n)
    Add all of these final results together.
    '''
    ss = sum((x - c) ** n for x in data)
    return ss


def stddev(data):
    """Standard Deviation"""
    # Get count/length of dataset
    n = len(data)
    # Get mean of our dataset
    c = mean(data)
    # Get sum of square deviations of sequence data
    ss = _ss(data, 2)
    '''
    Divide the sum of square deviations (ss) by the count/length of dataset (n-1)
    Get the square root of the previous result.
    '''
    return (ss / (n - 1)) ** 0.5


def skewness(data):
    """ 3'rd Moment, Skewness based on Formula """
    # Get count/length of dataset
    n = len(data)
    # Get mean of our dataset
    c = mean(data)
    # Get sum of cubic deviations of sequence data
    sc = _ss(data, 3)
    # Get Standard Deviation
    sd = stddev(data)
    '''
    Divide the sum of square deviations (sc) by: 
    The count/length of dataset (n-1), multiplied by the standard deviation (sd) to the power of 3, 
    due to it being cubic.
    '''
    return (sc / ((n - 1) * (sd ** 3)))


def kurtosis(data):
    """ 4'th Moment, Kurtosis based on Formula """
    # Get count/length of dataset
    n = len(data)
    # Get mean of our dataset
    c = mean(data)
    # Get sum of quadratic deviations of sequence data
    sq = _ss(data, 4)
    '''
    Divide the sum of square deviations (sc) by: 
    The count/length of dataset (n-1), multiplied by the standard deviation (sd) to the power of 4, 
    due to it being quadratic.
    Subtract 3 from the final result. The reason not to subtract off 3 is that the bare fourth moment 
    better generalizes to multivariate distributions, especially when independence is not assumed. 
    '''
    sd = stddev(data)
    return (sq / ((n - 1) * (sd ** 4))) - 3


# Function created to perform the above. Returning the cumulative return to plot, given the
# DataFrame dictionary, weights, codes, and the initial balance
# We'll be using this on the test data
def GenerateCumulativeReturns(df, weights, codes, balance):
    weighted_returns = pd.DataFrame()
    for asset in codes:
        weighted_returns[str(asset)] = (df[str(asset)]['Daily Return'] * weights[str(asset)][0])
    port_ret = weighted_returns.sum(axis=1)  # axis = 1, means count rows
    cumulative_ret = (balance - 1) + (port_ret + 1).cumprod()
    return cumulative_ret

def GeneratePortfolioReturns(df, weights, codes):
    weighted_returns = pd.DataFrame()
    for asset in codes:
        weighted_returns[str(asset)] = (df[str(asset)]['Daily Return'] * weights[str(asset)][0])
    port_ret = weighted_returns.sum(axis=1)  # axis = 1, means count rows
    return port_ret


# Optimization Functions =============================================================
def PortfolioDiversityRatio(data_dict, weights, codes):
    # Group Log returns together in one dataframe and get covariance.
    _df = pd.DataFrame()
    _vol_arr = []
    for code in codes:
        _df[str(code)] = data_dict[str(code)]['Log Return']
        _vol_arr.append(data_dict[str(code)]['Std'][0])
    covMatrix = _df.cov()
    diversity_arr = []
    for w in weights:  # list of weight arrays
        w_vol = np.dot(np.sqrt(_vol_arr), w.T)
        port_vol = np.dot(w.T, np.dot(covMatrix, w))
        diversity_arr.append((w_vol / math.sqrt(port_vol)))
    return diversity_arr


def PortfolioVolatility(data_dict, weights, codes):
    # Group Log returns together in one dataframe and get covariance.
    _df = pd.DataFrame()
    for code in codes:
        _df[str(code)] = data_dict[str(code)]['Log Return']

    covMatrix = _df.cov()
    volatility_arr = []
    for w in weights:
        volatility_arr.append(np.dot(w.T, np.dot(covMatrix, w)))
    return volatility_arr


# Creating custom functions for Expected return and Volatility
# weights assumed as DataFrame.
# Weights and codes in same order.
def PortfolioExpectedReturn(data_dict, weights, codes):
    er = []
    for w in weights:
        ExpectedReturn = 0
        for i in range(len(codes)):
            ExpectedReturn += (mean(data_dict[str(codes[i])]['Log Return']) * w[i])
        er.append(ExpectedReturn)
    return er
