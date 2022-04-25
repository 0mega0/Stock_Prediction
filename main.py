import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from My_LSTM_Training import My_LSTM_Training

 


def my_visualising(real_price, predict_price):
    # 可视化预测结果
    n = len(predict_price)
    length = len(predict_price) if n > 160 else 160
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.rcParams['figure.figsize'] = (length / 20, 6)
    plt.plot(real_price, color='red', label='实际股票价格走势')#, marker='o', markerfacecolor='red')  # 红线表示真实股价
    plt.plot(predict_price, color='blue', label='预测股票价格走势')#, marker='o', markerfacecolor='blue')  # 蓝线表示预测股价
    plt.title(' 股票走势预测')
    plt.xlabel('时间(天)')
    plt.ylabel('开盘单股价格(美元)')
    plt.legend()
    plt.show()



df = pd.read_csv('dataset_origin/TSLA.csv')
df = df['Open'].values
df = df.reshape(-1, 1)
dataset_train = np.array(df[:int(df.shape[0] * 0.8)])
dataset_test = np.array(df[int(df.shape[0] * 0.8):])
scaler = MinMaxScaler(feature_range=(0, 1))
dataset_train = scaler.fit_transform(dataset_train)
dataset_test = scaler.transform(dataset_test)
x_train, y_train = create_dataset(dataset_train)
x_test, y_test = create_dataset(dataset_test)
# print(len(dataset_test))
print(len(x_test))
print(len(y_test))
# visualising(real_price=y_test,predict_price=x_test)