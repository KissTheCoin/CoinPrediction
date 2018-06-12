# coding: utf-8
# Author: Ross
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
from keras.layers import LSTM, Dropout, Dense


def build_model(inputs, output_size, LSTM_units, activ_func='linear', dropout=0.1, loss='mae', optimizer='adam'):
    '''
    初始模型
    建立简单的LSTM的时序模型，由一层 LSTM和一层全连接层完成
    :param inputs: 输入 [a, b, c]
    :param output_size: 输出维度
    :param LSTM_units: LSTM单元数
    :param activ_func: 激活函数
    :param dropout: dropout率
    :param loss: 损失函数
    :param optimizer: 优化方法
    :return:
    '''
    print(inputs.shape[1])
    print(inputs.shape[2])
    model = Sequential()
    model.add(
        layers.LSTM(LSTM_units, input_shape=(inputs.shape[1], inputs.shape[2]), activation='tanh',
                    return_sequences=True))
    model.add(layers.Flatten())
    model.add(layers.Dense(units=100, kernel_initializer='normal', activation='relu'))
    model.add(layers.Activation(activ_func))
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(1, kernel_initializer='normal'))
    model.compile(optimizer=optimizer, loss=loss)
    return model


def build_model2(layers):
    d = 0.2
    model = Sequential()
    model.add(LSTM(128, input_shape=(layers.shape[1], layers.shape[2]), return_sequences=True))
    model.add(Dropout(d))
    model.add(LSTM(64, input_shape=(layers.shape[1], layers.shape[2]), return_sequences=False))
    model.add(Dropout(d))
    model.add(Dense(16, init='uniform', activation='relu'))
    model.add(Dense(1, init='uniform', activation='relu'))
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    return model
