from My_LSTM_Training import My_LSTM_Training


Timestep = 64
DATA_PATH = 'dataset_origin/'
MODEL_PATH = 'trained_models/'
myModel = My_LSTM_Training(Timestep, DATA_PATH, MODEL_PATH)
X_train, y_train = myModel.init_dataset('JD.csv')
# myModel.modelTrain(X_train, y_train, name='JD_e32b32tanh_2.h5')
# 开始训练模型
real_stock_price, predicted_stock_price = myModel.model_predict('JD+FB_e16b32relu.h5', 'JD.csv', Timestep, 10)
myModel.visualising(real_stock_price, predicted_stock_price)
