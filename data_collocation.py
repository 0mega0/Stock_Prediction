import tushare as ts
import pandas as pd
import yfinance as yf
def yfinance(stock_code, path):
    a = yf.Ticker(stock_code)
    df = a.history(period='max')
    df.to_csv(path, sep=',', encoding='utf-8')
# yfinance('AAPL', 'out.csv')
# 雅虎财经因为不再为大陆用户提供数据，所以暂时不能使用
# 要用的话把代码拷到google colab上去跑

def tushare(path,stock_code, start_date, end_date):
    pro = ts.pro_api('711feb83aee69bfac89a056c9d77b9776a61dd306222bddb9b9d09f7')
    df = pro.daily(ts_code=stock_code, start_date=start_date, end_date=end_date)
    # print(df)
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    #     print(df)
    df.to_csv(path+stock_code+'1.csv', sep=',', encoding='utf-8')
tushare('dataset_origin/CN/','00001.HK', '19000101', '20221231')




