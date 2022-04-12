import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from matplotlib.pyplot import figure
from datetime import datetime


class My_LSTM_Training:
    Timestep = 64  # 模型通过前*Timestep*天的内容进行预测
    DATA_PATH = ''
    MODEL_PATH = ''
    PIC_PATH = ''
    scaler = ''
    now_predict_file = ''
    now_model_name = ''

    def __init__(self, timestep, data_path, model_path, pic_path):
        self.Timestep = timestep
        self.DATA_PATH = data_path
        self.MODEL_PATH = model_path
        self.PIC_PATH = pic_path
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def create_dataset(self, data, timestep):
        # 用于辅助初始化数据的方法
        X_train = []  # 预测点前*Timestep* 的资料
        y_train = []  # 预测点
        for i in range(timestep, data.shape[0]):
            X_train.append(data[i - timestep:i, 0])
            y_train.append(data[i, 0])
        X_train, y_train = np.array(X_train), np.array(y_train)  # 转换成numpy array格式，方便输入到模型
        return X_train, y_train

    def init_dataset(self, data_file):
        origin_dataset_train = pd.read_csv(self.DATA_PATH + data_file)
        training_set = origin_dataset_train['Open'].values
        training_set = np.array(training_set.reshape(-1, 1))

        training_set_scaled = self.scaler.fit_transform(training_set)
        X_train, y_train = self.create_dataset(training_set_scaled, self.Timestep)  # 获取初始化数据
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        # 把X_train从2维数据reshape成三维数据[stock prices, timesteps, indicators]
        return X_train, y_train

    def modelTrain(self, X_train, y_train, epochs=32, batch_size=32, name='stock_prediction.h5'):
        model_train = Sequential()
        model_train.add(LSTM(units=64, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model_train.add(Dropout(0.12))
        model_train.add(LSTM(units=64, activation='relu', return_sequences=True))
        model_train.add(Dropout(0.12))
        model_train.add(LSTM(units=64, activation='relu'))
        model_train.add(Dropout(0.12))
        # 全连接，输出， add output layer
        model_train.add(Dense(units=1))
        model_train.compile(loss='mean_squared_error', optimizer='adam')
        model_train.summary()
        model_train.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
        # epochs: 训练轮数
        # batch_size: 一次训练所抓取的数据样本数量

        model_train.save(self.MODEL_PATH + name)

    def model_predict(self, model_name, data_predict, timestep, predict_days):
        self.now_model_name = model_name
        self.now_predict_file = data_predict[:-4]
        model = load_model(self.MODEL_PATH + model_name)
        origin_dataset_test = pd.read_csv(self.DATA_PATH + data_predict)
        real_stock_price = origin_dataset_test['Open'].values
        # 股市数据的实际数据(全部)
        real_stock_price_part = real_stock_price[len(real_stock_price) - predict_days:len(real_stock_price)]
        # 股市数据的实际数据(从后往前分割过的)
        dataset_test = origin_dataset_test['Open'].values
        dataset_test_part = dataset_test[len(dataset_test) - predict_days - timestep:len(dataset_test)]

        dataset_test_part = np.array(dataset_test_part.reshape(-1, 1))
        dataset_test_part = self.scaler.fit_transform(dataset_test_part)
        X_test = []
        for i in range(self.Timestep, dataset_test_part.shape[0]):
            X_test.append(dataset_test_part[i - self.Timestep + 1:i + 1, 0])
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        predicted_price = model.predict(X_test)
        predicted_price = self.scaler.inverse_transform(predicted_price)
        return real_stock_price_part, predicted_price

    def visualising(self, real_price, predict_price, if_generate_pic=False):
        # 可视化预测结果
        n = len(predict_price)
        length = len(predict_price) if n > 160 else 160
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        plt.rcParams['figure.figsize'] = (length / 20, 6)
        plt.plot(real_price, color='red', label='实际股票价格走势', marker='o', markerfacecolor='red')  # 红线表示真实股价
        plt.plot(predict_price, color='blue', label='预测股票价格走势', marker='o', markerfacecolor='blue')  # 蓝线表示预测股价
        plt.title(self.now_predict_file + ' 股票走势预测')
        plt.xlabel('时间(天)')
        plt.ylabel('开盘单股价格(美元)')
        plt.legend()
        if if_generate_pic:
            plt.savefig(self.PIC_PATH + self.now_model_name + '+' + self.now_predict_file + '.png')
            plt.clf()
        else:
            plt.show()

    def analysis(self, real_price_set, predict_price_set):
        accuracy = 0.0

# Timestep = 64
# DATA_PATH = 'dataset_origin/'
# MODEL_PATH = 'trained_models/'
# PIC_PATH = 'predict_pic/'
# myModel = My_LSTM_Training(Timestep, DATA_PATH, MODEL_PATH, PIC_PATH)
# X_train, y_train = myModel.init_dataset('train_set.csv')
# # myModel.modelTrain(X_train, y_train, batch_size=128, name='train_set_test.h5')
# # 开始训练模型
# real_stock_price, predicted_stock_price = myModel.model_predict('train_set_test.h5', 'JD.csv', Timestep, 50)
# myModel.visualising(real_stock_price, predicted_stock_price)
