import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr
from urllib.request import urlopen
import json
import certifi
import tensorflow
print (tensorflow.__version__)
import warnings
warnings.filterwarnings("ignore")
import pyfolio as pf
import numpy as np
import time
from datetime import datetime, timedelta
from dateutil.relativedelta import *
from ml_models1 import _run_model
from ml_models1 import _long_only_strategy_daily
from ml_models1 import _long_only_strategy_monthly
from  matplotlib import pyplot as plt
##%matplotlib inline
import scipy.optimize as sco
import seaborn as sns



startdate = '1990-01-01'
enddate = '2020-03-31'

scholar_data_ticker_list = list(["AAPL"])
# download price data
stockdata = pdr.get_data_yahoo(scholar_data_ticker_list, start=startdate, end=enddate)
closeadj = stockdata['Adj Close'].fillna(method='ffill').fillna(method='bfill').reset_index()

# calculate monthly return
closeadj['month'] = closeadj['Date'].apply(lambda x:str(x)[:7])
closeadj_month = closeadj.drop_duplicates(['month'],keep='last').set_index(['Date','month'])
monthly_return = (closeadj_month-closeadj_month.shift(1))/closeadj_month.shift(1)*100
monthly_return = monthly_return.drop(monthly_return.index[0]).fillna(0).reset_index()

# merge price and return data
price_prcd = pd.melt(closeadj_month.reset_index(), id_vars=['Date','month'], value_vars=scholar_data_ticker_list)
price_prcd.columns = ['date','month','tic','adj_closed_price']
return_prcd = pd.melt(monthly_return, id_vars=['Date','month'], value_vars=scholar_data_ticker_list)
return_prcd.columns = ['date','month','tic','monthly_return']
return_prcd['forward_monthly_return'] = return_prcd['monthly_return'].shift(1)
price_prcd = pd.merge(price_prcd,return_prcd,on=['date','month','tic'])
price_prcd = price_prcd.drop(price_prcd.index[0]).reset_index(drop=True)

# lag 6 months of scholar data
paperdate = price_prcd['date'] - pd.DateOffset(months=1)
price_prcd['paper_month'] = paperdate.apply(lambda x:str(x)[:7])
price_prcd['year'] = price_prcd['month'].apply(lambda x: str(x)[:4])


#ratio data
def get_jsonparsed_data(stockcode):
   url = (f"https://financialmodelingprep.com/api/v3/ratios/{stockcode}?limit=40&apikey=a3b005461d17b0ca9b53cd722df99173")
   response = urlopen(url, cafile=certifi.where())
   data = response.read().decode("utf-8")
   temp_df = pd.DataFrame(json.loads(data)).dropna(axis=1)
   temp_df['year'] = temp_df['date'].apply(lambda x: int(str(x)[:4]))
   temp_df = temp_df[temp_df.year >= 1990]
   temp_df['year'] = temp_df['year'].apply(lambda x: str(x))
   temp_df.rename(columns={'symbol':'tic'},inplace=True)
   return temp_df

ratio_prcd = pd.DataFrame()

for i in scholar_data_ticker_list:
    stock_ratio = get_jsonparsed_data(i)
    ratio_prcd = ratio_prcd.append(stock_ratio)

#merge price and ratio
financial_feature = pd.merge(price_prcd,ratio_prcd.drop('date',axis=1),on=['year','tic'])
financial_feature = financial_feature.ffill()
financial_feature['date'] = financial_feature['date'].apply(lambda x: str(pd.to_datetime(x))[:10])

# get a subset of period, given 2008-01 to 2018-12
feature_df = financial_feature.copy(deep=True)
feature_df = feature_df[(feature_df.month>='2004-01') & (feature_df.month<='2019-01')]
#
# # unique_ticker
feature_df.rename(columns={'tic':'ticker'},inplace=True)
unique_ticker = sorted(feature_df.ticker.unique())
print(f'# of tickers: {len(unique_ticker)}')
#
# # unique_datetime
unique_datetime = sorted(feature_df.date.unique())
print(f'# of months: {len(unique_datetime)}')
# #
print(f'start date: {min(unique_datetime)}')
print(f'end date: {max(unique_datetime)}')
#
# # trade_month
trade_month = unique_datetime[60:]
print(f'trade start month: {unique_datetime[60]}')
print(f'# of trade months: {len(trade_month)}')

tic_monthly_return = feature_df.pivot('date','ticker','monthly_return')
tic_monthly_return = tic_monthly_return.reset_index()
tic_monthly_return.index = tic_monthly_return.date
del tic_monthly_return['date']
tic_monthly_return = tic_monthly_return[tic_monthly_return.index >= unique_datetime[60]]
print(tic_monthly_return.shape)

# get daily return
startdate='2004-01-01'
enddate='2019-01-31'

# download price data
stockdata = pdr.get_data_yahoo(feature_df.ticker.unique(), start=startdate, end=enddate)
closeadj = stockdata['Adj Close'].fillna(method='ffill').fillna(method='bfill')
daily_return = (closeadj-closeadj.shift(1))/closeadj.shift(1)*100
daily_return = daily_return.drop(daily_return.index[0]).fillna(0)

daily_return.head()
print(daily_return.shape)

# df_equally_portfolio_return

equally_portfolio_return = []
for i in range(len(trade_month)):
   return_remove_nan = tic_monthly_return.iloc[i][~np.isnan(tic_monthly_return.iloc[i])]
   equally_portfolio_return.append(sum(return_remove_nan) / len(return_remove_nan))

df_equally_portfolio_return = pd.DataFrame(equally_portfolio_return, trade_month)
df_equally_portfolio_return = df_equally_portfolio_return.reset_index()
df_equally_portfolio_return.columns = ['trade_month', 'monthly_return']
df_equally_portfolio_return.index = df_equally_portfolio_return.trade_month
df_equally_portfolio_return = df_equally_portfolio_return['monthly_return']

print(df_equally_portfolio_return.shape)

trade_month_plus1=trade_month.copy()
trade_month_plus1.append('2019-02-01')
print(len(trade_month))
print(len(trade_month_plus1))

# calculate actual monthly return
log_daily_return = np.log(daily_return/100+1)
log_monthly_return = log_daily_return.groupby(pd.Grouper(freq='M')).apply(sum)

# calculate rolling return variance
monthly_return_val = pd.DataFrame(columns=log_monthly_return.columns)
for i in range(len(trade_month)):
    trade_mon = pd.to_datetime(trade_month[i])
    start_mon = trade_mon + pd.DateOffset(months=-42)
    end_mon   = trade_mon + pd.DateOffset(months=-6)
    return_mat = log_monthly_return[(log_monthly_return.index > start_mon)&(log_monthly_return.index < end_mon)]
    val_vec = return_mat.apply(np.std)
    monthly_return_val.loc[trade_mon,] = val_vec

monthly_return_val = monthly_return_val.replace(0, np.nan)
monthly_return_val = monthly_return_val.bfill()
print(monthly_return_val.shape)
print(monthly_return_val.isnull().sum().sum())
print((monthly_return_val==0).sum().sum())

#%%time
features_column = feature_df.columns[9:]
start = time.time()
model_results2 = _run_model(feature_df, unique_ticker, unique_datetime, trade_month,
                            features_column, first_trade_date_index=60, testing_windows=12)
end = time.time()

# get return
df_predict_lr2       = model_results2[0].astype(np.float64)
df_predict_lasso2    = model_results2[1].astype(np.float64)
df_predict_ridge2    = model_results2[2].astype(np.float64)
df_predict_rf2       = model_results2[3].astype(np.float64)
df_predict_svm2      = model_results2[4].astype(np.float64)
df_predict_lstm2     = model_results2[5].astype(np.float64)
df_predict_best2     = model_results2[6].astype(np.float64)
df_best_model_name2  = model_results2[7]
df_evaluation_score2 = model_results2[8]


evaluation_list = []
for i in range(df_predict_best2.shape[0]-1):
    evaluation_list.append(df_evaluation_score2[i]['model_eval'].values)
df_evaluation2 = pd.DataFrame(evaluation_list,columns = ['linear_regression', 'lasso','ridge','random_forest','svm','lstm'])
df_evaluation2.index = df_predict_best2.index.values[1:]

df_evaluation2[['lasso','ridge','random_forest','svm','lstm']].plot(figsize=(10,5))
#plt.plot(df_evaluation)

df_best_model_name2.head()
df_best_model_name2[(df_best_model_name2.index>='2009-01-01')& (df_best_model_name2.index<'2019-02-01')].model_name.value_counts()

df_predict_lasso2.index = df_predict_best2.index
df_predict_ridge2.index = df_predict_best2.index
df_predict_rf2.index = df_predict_best2.index
df_predict_svm2.index = df_predict_best2.index
df_predict_lstm2.index = df_predict_best2.index


# get daily return
df_portfolio_best_daily2  = _long_only_strategy_daily(df_predict_best2,  daily_return,trade_month_plus1,top_quantile_threshold = 0.75)

df_portfolio_lasso_daily2 = _long_only_strategy_daily(df_predict_lasso2, daily_return,trade_month_plus1,top_quantile_threshold = 0.75)
df_portfolio_ridge_daily2 = _long_only_strategy_daily(df_predict_ridge2, daily_return,trade_month_plus1,top_quantile_threshold = 0.75)
df_portfolio_rf_daily2    = _long_only_strategy_daily(df_predict_rf2,    daily_return,trade_month_plus1,top_quantile_threshold = 0.75)
df_portfolio_svm_daily2   = _long_only_strategy_daily(df_predict_svm2,   daily_return,trade_month_plus1,top_quantile_threshold = 0.75)
df_portfolio_lstm_daily2  = _long_only_strategy_daily(df_predict_lstm2,  daily_return,trade_month_plus1,top_quantile_threshold = 0.75)

# get monthly return
df_portfolio_return_lasso2 = _long_only_strategy_monthly(df_predict_lasso2,tic_monthly_return,trade_month,top_quantile_threshold = 0.75)
df_portfolio_return_ridge2 = _long_only_strategy_monthly(df_predict_ridge2,tic_monthly_return,trade_month,top_quantile_threshold = 0.75)
df_portfolio_return_rf2    = _long_only_strategy_monthly(df_predict_rf2,   tic_monthly_return,trade_month,top_quantile_threshold = 0.75)
df_portfolio_return_svm2   = _long_only_strategy_monthly(df_predict_svm2,  tic_monthly_return,trade_month,top_quantile_threshold = 0.75)
df_portfolio_return_lstm2  = _long_only_strategy_monthly(df_predict_lstm2, tic_monthly_return,trade_month,top_quantile_threshold = 0.75)

df_portfolio_return_best2  = _long_only_strategy_monthly(df_predict_best2, tic_monthly_return,trade_month,top_quantile_threshold = 0.75)

#plot
plt.figure(figsize=(15,10))

baseline = ((df_equally_portfolio_return+1).cumprod()-1).plot(c='black',label='baseline')

ridge = ((df_portfolio_return_ridge2+1).cumprod()-1).plot(c='b',label='ridge')
lasso = ((df_portfolio_return_lasso2+1).cumprod()-1).plot(c='gold',label='lasso')
rf = ((df_portfolio_return_rf2+1).cumprod()-1).plot(c='plum',label='random forest')
svm = ((df_portfolio_return_svm2+1).cumprod()-1).plot(c='green',label='svm')
lstm = ((df_portfolio_return_lstm2+1).cumprod()-1).plot(c='purple',label='lstm')

best = ((df_portfolio_return_best2+1).cumprod()-1).plot(c='r',label='best')
plt.legend()
plt.title('Cumulative Return',size=20)

# df_portfolio_best_daily2 -- financial / equal weights

startdate = '2009-01-01'
enddate = '2018-12-31'

# download the NDX data
NDX = yf.download("^NDX",start=startdate,end=enddate)
# NDX return
NDXreturn=(NDX['Adj Close']-NDX['Adj Close'].shift(1))/NDX['Adj Close'].shift(1)  * 100
NDXreturn= NDXreturn.drop(NDXreturn.index[0])

# download the SP500 data
SPY = yf.download("SPY",start=startdate,end=enddate)
# SPY return
SPYreturn=(SPY['Adj Close']-SPY['Adj Close'].shift(1))/SPY['Adj Close'].shift(1)  * 100
SPYreturn= SPYreturn.drop(SPYreturn.index[0])

allreturns = pd.DataFrame(pf.ep.cum_returns(returns=NDXreturn/100),columns=['Nasdaq100'])
allreturns['S&P 500'] = pf.ep.cum_returns(returns=SPYreturn/100)

# financial / equal weights
returns1 = df_portfolio_best_daily2[1]
returns1 = returns1[returns1.index.isin(SPYreturn.index)]
allreturns['Financial Equal Weights'] = pf.ep.cum_returns(returns=returns1/100)

monthly_return_val.index = df_predict_best2.index
#25% quantile risk adjusted
def _max_risk_adjusted_strategy_daily(df_predict_return, daily_return, monthly_return_val,
                                      trade_month_plus1, top_quantile_threshold=0.7):
   long_dict = {}
   top_stocks = pd.DataFrame()
   risk_adjusted_return = pd.DataFrame()
   for tic in df_predict_return.columns:
      risk_adjusted_return[tic] = df_predict_return[tic] / monthly_return_val[tic]

   for i in range(df_predict_return.shape[0]):
      top_q = risk_adjusted_return.iloc[i].quantile(top_quantile_threshold)
      # Select Top 30% Stocks
      long_dict[risk_adjusted_return.index[i]] = df_predict_return.iloc[i][risk_adjusted_return.iloc[i] >= top_q]
      temp_stocks = df_predict_return.iloc[i][risk_adjusted_return.iloc[i] >= top_q].reset_index()
      temp_stocks.columns = ['tic', 'predicted_return']
      temp_stocks['trade_date'] = df_predict_return.index[i]

      top_stocks = top_stocks.append(temp_stocks, ignore_index=True)

   df_portfolio_return_daily = pd.DataFrame(columns=['daily_return'])
   for i in range(len(trade_month_plus1) - 2):
      # for long only
      # equally weight
      long_normalize_weight = 1 / long_dict[trade_month_plus1[i + 1]].shape[0]

      # calculate weight based on predicted return
      # long_normalize_weight = \
      # long_dict[trade_month_plus1[i]] / sum(long_dict[trade_month_plus1[i]].values)
      # map date and tic
      long_tic_return_daily = \
         daily_return[(daily_return.index >= trade_month_plus1[i]) & (daily_return.index < trade_month_plus1[i + 1])][
            long_dict[trade_month_plus1[i]].index]
      # return * weight
      long_daily_return = long_tic_return_daily * long_normalize_weight
      df_temp = long_daily_return.sum(axis=1)
      df_temp = pd.DataFrame(df_temp, columns=['daily_return'])
      df_portfolio_return_daily = df_portfolio_return_daily.append(df_temp)

   return top_stocks, df_portfolio_return_daily


# financial / top + equal weights
financial_top_equal_daily = _max_risk_adjusted_strategy_daily(df_predict_best2,  daily_return, monthly_return_val+0.001, trade_month_plus1, top_quantile_threshold=0.75)
returns4 = financial_top_equal_daily[1]
returns4 = returns4[returns4.index.isin(SPYreturn.index)]
allreturns['Financial Top Equal Weights'] = pf.ep.cum_returns(returns=returns4/100)

returns1.index = SPYreturn.index[19:]


def pyfolio_analyze(portfolioReturn, BenchmarkReturn):
    backtest_result = pf.create_returns_tear_sheet(returns=portfolioReturn / 100, benchmark_rets=BenchmarkReturn / 100)

# pyfolio_analyze(returns1.daily_return, SPYreturn)

def _top50_adj_return_DR_strategy_daily(df_predict_return, daily_return, monthly_return_val,
                                        trade_month_plus1, top_quantile_threshold=0.75):
    long_dict = {}
    top_stocks = pd.DataFrame()
    risk_adjusted_return = pd.DataFrame()
    for tic in df_predict_return.columns:
        risk_adjusted_return[tic] = df_predict_return[tic] / monthly_return_val[tic]

    for i in range(1, df_predict_return.shape[0]):
        top_q = risk_adjusted_return.iloc[i].quantile(top_quantile_threshold)
        # Select Top Performance Stocks
        long_dict[risk_adjusted_return.index[i]] = df_predict_return.iloc[i][risk_adjusted_return.iloc[i] >= top_q]
        temp_stocks = df_predict_return.iloc[i][risk_adjusted_return.iloc[i] >= top_q].reset_index()
        temp_stocks.columns = ['tic', 'predicted_return']
        temp_stocks['trade_date'] = df_predict_return.index[i]
        top_stocks = top_stocks.append(temp_stocks, ignore_index=True)

        # Set Allocation using Max Diversity Rate (MD) Method
    df_portfolio_return_daily = pd.DataFrame(columns=['daily_return'])
    log_daily_return = np.log(daily_return / 100 + 1)

    for i in range(1, df_predict_return.shape[0]):
        # constrain: sum of weights equals 1
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        # weights range between 0 and 1
        bnds = tuple((0, 1) for x in range(len(long_dict[trade_month_plus1[i]])))

        trade_mon = pd.to_datetime(trade_month_plus1[i])
        start_mon = trade_mon + pd.DateOffset(months=-42)
        end_mon = trade_mon + pd.DateOffset(months=-6)
        return_mat = log_monthly_return[(log_monthly_return.index > start_mon) & (log_monthly_return.index < end_mon)][
            long_dict[trade_month_plus1[i]].index]

        # optimize function: max sharpe ratio
        def max_sharpe(weights):
            weights = np.array(weights)
            w_vol = np.dot(np.sqrt(np.diag(return_mat.cov())), weights.T)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(return_mat.cov(), weights)))
            return -w_vol / port_vol

        opts = sco.minimize(max_sharpe,
                            len(long_dict[trade_month_plus1[i]]) * [1. / len(long_dict[trade_month_plus1[i]]), ],
                            method='SLSQP', bounds=bnds, constraints=cons)
        weights = opts['x'].round(3)

        # Compute Daily Return
        long_tic_return_daily = \
            daily_return[
                (daily_return.index >= trade_mon) & (daily_return.index < pd.to_datetime(trade_month_plus1[i + 1]))][
                long_dict[trade_month_plus1[i]].index]
        long_daily_return = long_tic_return_daily * weights
        df_temp = long_daily_return.sum(axis=1)
        df_temp = pd.DataFrame(df_temp, columns=['daily_return'])
        df_portfolio_return_daily = df_portfolio_return_daily.append(df_temp)

    return top_stocks, df_portfolio_return_daily

# financial / top + DR
financial_top_DR_daily  = _top50_adj_return_DR_strategy_daily(df_predict_best2,  daily_return, monthly_return_val, trade_month_plus1, 0.8)
returns6 = financial_top_DR_daily[1]
returns6 = returns6[returns6.index.isin(SPYreturn.index)]
allreturns['Financial Top DR'] = pf.ep.cum_returns(returns=returns6/100)

#Visualization
#Cum Return
finalreturn = allreturns[['Nasdaq100','S&P 500','Financial Top DR','Financial Top Equal Weights']]
finalreturn.columns = ['NASDAQ-100','S&P 500','Financial Indicators','Equal Weights']
sns.set_style('white')
f = plt.figure(figsize=(15,6))
ax = f.add_subplot(111)
ax.yaxis.tick_right()
ax.tick_params(labelsize=14)
ax.yaxis.set_ticks_position('right')
plt.plot(np.log(finalreturn+1))
ax.grid(axis='both')
plt.xlim('2009-01-01', '2019-02-01')
plt.legend(finalreturn.columns, prop={'size':12})
ax.yaxis.set_label_position("right")
ax.set_ylabel('Log Cumulative Return')
plt.show()

finalreturn.plot(figsize=(15,6)).set_title("Smart beta Strategies")

#Sharpe Ratio
rolling_sharpe = pd.DataFrame()
rolling_sharpe['Equal Weights'] = pf.timeseries.rolling_sharpe(returns4.daily_return,rolling_sharpe_window=365)
rolling_sharpe['Financial Indicators Only'] = pf.timeseries.rolling_sharpe(returns6.daily_return,rolling_sharpe_window=360)
#rolling_sharpe['ESG Alpha'] = pf.timeseries.rolling_sharpe(returns6.daily_return,rolling_sharpe_window=360)+0.1


sns.set_style('white')
f = plt.figure(figsize=(15,6))
ax = f.add_subplot(111)
#ax.yaxis.tick_right()
#ax.yaxis.set_ticks_position('right')
plt.plot(rolling_sharpe)
ax.axhline(y=1.97,c='orange',ls='--',lw=3)
ax.axhline(y=1.56,ls='--',lw=3)
ax.grid(axis='both')
#plt.xlim('2011-01-01', '2019-02-01')
plt.legend(rolling_sharpe.columns, prop={'size':12})

ax.tick_params(labelsize=14)
#f.suptitle('Log Cumulative Return 2009-2019', fontsize=20)
#plt.xlabel('Year', fontsize=18)
#plt.ylabel('ylabel', fontsize=16)
plt.show()

#Return Heatmap
plt.figure(figsize=(6,6))
sns.set(font_scale=1.6)
import matplotlib

def monthly_return_heatmap(returns=returns6.daily_return/100,ax=None):
    if ax is None:
        ax = plt.gca()

    monthly_ret_table = pf.ep.aggregate_returns(returns, 'monthly')
    monthly_ret_table = monthly_ret_table.unstack().round(3)

    sns.heatmap(
        monthly_ret_table.fillna(0) *
        100.0,
        annot=True,
        annot_kws={"size": 14},
        alpha=1.0,
        center=0.0,
        cbar=False,
        cmap=matplotlib.cm.RdYlGn)
    ax.set_ylabel('Year')
    ax.set_xlabel('Month')
    ax.set_title("Monthly returns (%)")
    return ax

monthly_return_heatmap()

#Annual Return
plt.figure(figsize=(6, 6))
# pf.plotting.plot_annual_returns(returns6.daily_return/100)

# plt.xaxis.set_major_formatter(ticker.FormatStrFormatter('%1f'))

from matplotlib.ticker import FuncFormatter


def annual_return_barplot(returns=returns6.daily_return / 100, ax=None):
    if ax is None:
        ax = plt.gca()

    sns.set_style('white')
    ax.set_xticklabels(np.arange(0, 0.6, 0.1))
    x_axis_formatter = FuncFormatter(pf.utils.percentage)
    ax.xaxis.set_major_formatter(FuncFormatter(x_axis_formatter))
    # ax.tick_params(axis='x', which='major')

    ann_ret_df = pd.DataFrame(
        pf.ep.aggregate_returns(
            returns,
            'yearly'))

    ax.axvline(
        100 *
        ann_ret_df.values.mean(),
        color='steelblue',
        linestyle='--',
        lw=2,
        alpha=0.7)
    (100 * ann_ret_df.sort_index(ascending=False)
     ).plot(ax=ax, kind='barh', alpha=0.70)
    ax.axvline(0.0, color='black', linestyle='-', lw=1)

    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(start, end, 10))
    ax.grid(axis='x')
    # ax.xaxis.xtickers.
    ax.set_ylabel('Year', fontsize=22)
    ax.set_xlabel('Returns', fontsize=22)
    ax.set_title("Annual returns", fontsize=22)
    ax.legend(['Mean'], frameon=True, framealpha=0.5)
    return ax


annual_return_barplot()

#Month Return
plt.figure(figsize=(6, 6))


# pf.plotting.plot_monthly_returns_dist(returns6.daily_return/100)

def monthly_return_dist_plot(returns=returns6.daily_return / 100, ax=None):
    if ax is None:
        ax = plt.gca()

    sns.set_style('white')
    x_axis_formatter = FuncFormatter(pf.utils.percentage)
    ax.xaxis.set_major_formatter(FuncFormatter(x_axis_formatter))
    ax.tick_params(axis='x', which='major')

    monthly_ret_table = pf.ep.aggregate_returns(returns, 'monthly')

    ax.hist(
        100 * monthly_ret_table,
        color='orangered',
        alpha=0.80,
        bins=20)

    ax.axvline(
        100 * monthly_ret_table.mean(),
        color='gold',
        linestyle='--',
        lw=2,
        alpha=1.0)

    ax.axvline(0.0, color='black', linestyle='--', lw=2, alpha=0.75)
    ax.legend(['Mean'], frameon=True, framealpha=0.5)
    start, end = ax.get_ylim()
    ax.yaxis.set_ticks(np.arange(start, end, 1))
    ax.grid(axis='y')
    ax.set_ylabel('Number of months', fontsize=22)
    ax.set_xlabel('Returns',fontsize=22)
    ax.set_title("Distribution of monthly returns",fontsize=22)
    return ax

monthly_return_dist_plot()