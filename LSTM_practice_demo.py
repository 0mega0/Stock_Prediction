from keras.models import Sequential, load_model
from My_LSTM_Training import My_LSTM_Training
import os


# def main1():
#     # myModel = My_LSTM_Training(64, 'dataset_origin/', 'trained_models/')
#     model = load_model('trained_models/JD_e32b32relu_2.h5')
#     X_train, y_train = myModel.init_dataset('FB.csv')
#     model.fit(X_train, y_train, epochs=16, batch_size=32)
#     model.save('trained_models/JD+FB_e16b32relu.h5')


def predict_multi(file):
    real_stock_price, predicted_stock_price = myModel.model_predict(model_name='train_set_test.h5', data_predict=file, timestep=64, predict_days=160)
    myModel.visualising(real_stock_price, predicted_stock_price, True)


Timestep = 64
DATA_PATH = 'dataset_origin/NYSE/small/'
MODEL_PATH = 'trained_models/'
PIC_PATH = 'predict_pic/NYSE.small/'
myModel = My_LSTM_Training(Timestep, DATA_PATH, MODEL_PATH, PIC_PATH)

for name in os.listdir(DATA_PATH):
    predict_multi(name)
