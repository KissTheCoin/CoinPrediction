# coding: utf-8
# Author: Ross

from util.coin_data import HistoryData
from model import LSTM
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

DATA_PATH = r'G:\project\Coin\CoinPrediction\data\eos-20130428-20180613.csv'
RANDOM_SEED = 188

if __name__ == '__main__':
    np.random.seed(RANDOM_SEED)  # 方便结果重现
    # 数据处理
    data = HistoryData(DATA_PATH)
    data.date2weekday().add_increase_col().drop_columns(['Date']).scalar(['Open*', 'High', 'Close**'], 0.01).split(0.8)
    train_x, train_y, test_x, test_y = data.generate_train_test_data(
        x_columns=['Open*', 'High', 'Close**', 'Weekday'],
        y_columns=['Close**'],
        window_len=7,
        flatten=False
    )
    print(train_x.shape)
    print(test_x.shape)
    print(train_y.shape)
    print(test_y.shape)

    # 模型训练
    model = LSTM.build_model(inputs=train_x, output_size=7, LSTM_units=20, loss='mean_squared_error')
    history = model.fit(train_x, train_y, batch_size=32, epochs=200, verbose=1, shuffle=True)

    # loss可视化
    fig, ax1 = plt.subplots(1, 1)
    ax1.plot(history.epoch, history.history['loss'])
    ax1.set_title('Training loss')
    ax1.set_ylabel('Mean Absolute Error (MAE)', fontsize=12)
    ax1.set_xlabel('# Epochs', fontsize=12)
    plt.show()

    # 预测结果可视化
    h = model.predict(test_x)
    h = h.reshape(len(h), -1)
    dat = np.concatenate((test_y, h), axis=1)
    df = pd.DataFrame(dat)
    df.to_csv('prediction.csv', index=False)
    plt.plot(np.mean(h, axis=1), color='blue', label='y_test')
    plt.plot(test_y[:, 1], color='red', label='prediction')
    plt.show()
