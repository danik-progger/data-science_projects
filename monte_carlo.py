import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from pandas_datareader import data as pdr
import yfinance as yf

yf.pdr_override()


# Preporation
def get_stocks_data(stocks, start, end):
    stock_data = pdr.get_data_yahoo(stocks, start, end)
    stock_data = stock_data['Close']
    returns = stock_data.pct_change()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    return mean_returns, cov_matrix


stock_list = ['CBA', 'BHP', 'TLS', 'NAB', 'WBC', 'STO']
stocks = [stock + '.AX' for stock in stock_list]
end_date = dt.datetime.now()
start_date = end_date - dt.timedelta(days=300)

# Create data
mean_returns, cov_matrix = get_stocks_data(stocks, start_date, end_date)
weights = np.random.random(len(mean_returns))
weights /= np.sum(weights)

# Monte Carlo simulation
simulations_number = 100
days_in_timeframe = 100
initial_portfolio = 1000

meanM = np.full(shape=(days_in_timeframe, len(weights)),
                fill_value=mean_returns).T
portfolio_simulations = np.full(shape=(days_in_timeframe, simulations_number),
                                fill_value=0.0)
for simulation in range(simulations_number):
    Z = np.random.normal(size=(days_in_timeframe, len(weights)))
    L = np.linalg.cholesky(cov_matrix)
    daily_returns = meanM + np.inner(L, Z)
    portfolio_simulations[:, simulation] = np.cumprod(
        np.inner(weights, daily_returns.T) + 1) * initial_portfolio

plt.plot(portfolio_simulations)
plt.title('Monte Carlo simulation of stocks portfolio')
plt.xlabel('Days')
plt.ylabel('Value ($)')
plt.show()
