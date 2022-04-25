import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from datetime import datetime


class My_LSTM_Training:
    Timestep = 64  # 模型通过前*Timestep*天的内容进行预测
    DATA_PATH = ''
    MODEL_PATH = ''
    PIC_PATH = ''
    now_predict_file = ''
    now_model_name = ''
    scaler = MinMaxScaler(feature_range=(0, 1))

    def __init__(self, timestep, data_path, model_path, pic_path):
        self.Timestep = timestep
        self.DATA_PATH = data_path
        self.MODEL_PATH = model_path
        self.PIC_PATH = pic_path

    def create_dataset(self, data, timestep):
        # 用于辅助初始化数据的方法
        x = []  # 预测点前*Timestep* 的资料
        y = []  # 预测点
        for i in range(timestep, data.shape[0]):
            x.append(data[i - timestep:i, 0])
            y.append(data[i, 0])
        x, y = np.array(x), np.array(y)
        # 转换成numpy array格式，方便输入到模型

        '''
        输出的x为: 一个数组,数组内容为(data长度 - timestep)个数组,数组的长度为timestep
        用于后续预测. 通过前timestep个数,预测后一个数的内容
        输出的y为: 一个数组,数组内容为(data长度 - timestep)个数
        实则为原data数据的划分data[:-timestep],即data去掉前timestep个元素后,剩下的部分
        '''
        return x, y

    def init_dataset(self, data_file):
        origin_dataset_train = pd.read_csv(self.DATA_PATH + data_file)
        # 从CSV文件中读取数据
        training_set = origin_dataset_train['Open'].values
        # 从数据集文件中，取出Open(开盘单股价格)列作为模型训练的数据。

        training_set = np.array(training_set.reshape(-1, 1))
        # 将数据转换成numpy array格式，方便输入到模型
        
        dataset_train = np.array(training_set[:int(training_set.shape[0] * 0.6)])
        dataset_test = np.array(training_set[int(training_set.shape[0] * 0.6):])
        # 将原始数据划分为训练集和测试集

        dataset_train_scaled = self.scaler.fit_transform(dataset_train)
        # 将训练集数据进行标准化处理
        dataset_test_scaled = self.scaler.transform(dataset_test)
        # 对测试集数据也进行标准化处理，这里的均值、方差、最大值最小值等等，等同于训练集的均值、方差、最大值最小值

        x_train, y_train = self.create_dataset(dataset_train_scaled, self.Timestep)
        x_test, y_test = self.create_dataset(dataset_test_scaled, self.Timestep)
        # 将数据通过create_dataset() 变成多个数据集合

        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        # 把x_train和x_test从二维数据reshape成三维数据

        return x_train, y_train, x_test, y_test

    def modelTrain(self, X_train, y_train, epochs=32, batch_size=32, name='stock_prediction.h5'):
        model_train = Sequential()
        model_train.add(LSTM(units=80, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model_train.add(Dropout(0.12))
        model_train.add(LSTM(units=80, activation='relu', return_sequences=True))
        model_train.add(Dropout(0.12))
        model_train.add(LSTM(units=80, activation='relu'))
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

        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset_test_part = scaler.fit_transform(dataset_test_part)
        X_test = []
        for i in range(self.Timestep, dataset_test_part.shape[0]):
            X_test.append(dataset_test_part[i - self.Timestep + 1:i + 1, 0])
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        predicted_price = model.predict(X_test)
        predicted_price = scaler.inverse_transform(predicted_price)
        # 还原为原始数据
        return real_stock_price_part, predicted_price

    def model_predict_from_list(self, model_name, data_predict, timestep, predict_days):
        if predict_days == '':
            predict_days = len(data_predict) - timestep

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

Timestep = 64
DATA_PATH = 'dataset_origin/'
MODEL_PATH = 'trained_models/'
PIC_PATH = 'predict_pic/'
myModel = My_LSTM_Training(Timestep, DATA_PATH, MODEL_PATH, PIC_PATH)
X_train, y_train = myModel.init_dataset('train_set.csv')
# # myModel.modelTrain(X_train, y_train, batch_size=128, name='train_set_test.h5')
# # 开始训练模型
# real_stock_price, predicted_stock_price = myModel.model_predict('train_set_test.h5', 'JD.csv', Timestep, 50)
# myModel.visualising(real_stock_price, predicted_stock_price)
