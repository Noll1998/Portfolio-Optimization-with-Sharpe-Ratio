import yfinance as yf
import numpy as np
from scipy.optimize import minimize

tickers = []
ticker = tickers.append(input("Add a stock to the list, when done type OK: "))

while True:
    ticker = input("Add a stock to the list, when done type OK: ")
    if ticker != "OK":
        tickers.append(ticker)
    else:
        break

start_date = input("Start date of analysis (format yyyy-mm-dd): ")
end_date = input("End date of analysis (format yyyy-mm-dd): ")

risk_free_rate = float(input("Risk free rate: "))

stocks = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
market = yf.download('^GSPC', start=start_date, end=end_date)['Adj Close']

returns = stocks.pct_change().dropna()
market_returns = market.pct_change().dropna()

betas = {}
for ticker in tickers:
    stock_returns = returns[ticker]
    covariance = stock_returns.cov(market_returns)
    variance = market_returns.var()
    beta = covariance / variance
    betas[ticker] = beta
    print(f'Beta for {ticker}: {beta}')

correlations = returns.corr()
print(f'Correlation between stocks:\n{correlations}')

expected_returns = []
for i, ticker in enumerate(tickers):
    stock_beta = betas[ticker]
    stock_expected_return = risk_free_rate + stock_beta * (market_returns.mean() - risk_free_rate)
    expected_returns.append(stock_expected_return)

best_combination = []

def objective(weights):
    portfolio_expected_return = np.dot(weights, expected_returns)
    if np.count_nonzero(weights) == 1:
        stock_risk = np.sqrt(returns[best_combination[0]].var())
        portfolio_risk = stock_risk
    else:
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(returns.cov(), weights)))
    sharpe_ratio = (portfolio_expected_return - risk_free_rate) / portfolio_risk
    return -sharpe_ratio

constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

bounds = tuple((0, 1) for i in range(len(tickers)))

initial_weights = [1 / len(tickers)] * len(tickers)
result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
best_weights = result.x

print("Best portfolio:")
for i, ticker in enumerate(tickers):
    if best_weights[i] > 0.001:
        best_combination.append(ticker)
        stock_returns = returns[ticker]
        stock_beta = betas[ticker]
        stock_expected_return = risk_free_rate + stock_beta * (market_returns.mean() - risk_free_rate)
        if len(best_combination) == 1: 
            stock_risk = np.sqrt(returns[best_combination[0]].var())
        else:
            stock_risk = np.sqrt(np.dot(best_weights.T, np.dot(returns.cov(), best_weights)))
        print(f'{ticker}: Weight: {best_weights[i]}, Expected return: {stock_expected_return}, Risk: {stock_risk}')

if len(best_combination) == 1:
    portfolio_risk = np.sqrt(returns[best_combination[0]].var())
else:
    portfolio_risk = np.sqrt(np.dot(best_weights.T, np.dot(returns.cov(), best_weights)))

portfolio_expected_return = np.dot(best_weights, expected_returns)
print(f'Best combination: {best_combination} with Sharpe ratio: {-(result.fun)}, Expected return: {portfolio_expected_return}, Risk: {portfolio_risk}')
