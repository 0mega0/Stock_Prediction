import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout


def create_dataset(data, timestep):
    X_train = []  # 预测点前*Timestep* 的资料
    y_train = []  # 预测点
    for i in range(timestep, data.rehsape[0]):
        X_train.append(data[i - timestep:i, 0])
        y_train.append(data[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)  # 转换成numpy array格式，方便输入模型
    return X_train, y_train


def modelTrain(x_train, y_train):
    model_train = Sequential()
    model_train.add(LSTM(units=64, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model_train.add(Dropout(0.12))
    model_train.add(LSTM(units=64, activation='relu', return_sequences=True))
    model_train.add(Dropout(0.12))
    model_train.add(LSTM(units=64, activation='relu'))
    model_train.add(Dropout(0.12))

    # 全连接，输出， add output layer
    model_train.add(Dense(units=1))
    model_train.compile(loss='mean_squared_error', optimizer='adam')
    model_train.summary()
    model_train.fit(x_train, y_train, epochs=16, batch_size=64)

    model_train.save('stock_prediction_trainset.h5')


def visualization(model, x_test):
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    fig, ax = plt.subplots(figsize=(100, 24))
    ax.set_facecolor('#000041')
    ax.plot(y_test_scaled, color='red', label='Original price')
    plt.plot(predictions, color='cyan', label='Predicted price')
    plt.title("JD prediction")
    plt.legend()
    plt.show()


df = pd.read_csv('dataset_origin/TSLA.csv')
# df = pd.read_csv('train_set.csv')
dfValidation = pd.read_csv('validation_set.csv')
dfTest = pd.read_csv('dataset_origin/JD.csv')
print(df.shape)
df = df['Open'].values
dfValidation = dfValidation['Open'].values
dfTest = dfTest['Open'].values

df = df.reshape(-1, 1)
dfValidation = dfValidation.reshape(-1, 1)
dfTest = dfTest.reshape(-1, 1)
dfTest = dfTest[int(dfTest[0] * 0.7):]
# df = df[:int(df.shape[0] * 0.7)]
dataset_train = np.array(df)
dataset_validation = np.array(dfValidation)
dataset_test = np.array(dfTest)
scaler = MinMaxScaler(feature_range=(0, 1))
dataset_train = scaler.fit_transform(dataset_train)
dataset_validation = scaler.fit_transform(dataset_validation)
dataset_test = scaler.transform(dataset_test)

x_train, y_train = create_dataset(dataset_train, 30)
# y_train = np.array(dataset_validation)
x_test, y_test = create_dataset(dataset_test, 30)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# modelTrain(x_train, y_train)
# model = load_model('stock_prediction_trainset.h5')
# # visualization(model, x_test)
# print(len(model.predict(x_test)[0]))
# plt.plot(model.predict(x_test)[0])
# plt.show()
