# coding: utf-8
# Author: Ross
import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class HistoryData:

    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)
        self.normalizer = None
        self.train = None
        self.test = None

    def date2weekday(self):
        """
        新增星期数，保存在 Weekday 列中
        :return: self
        """
        self.data['Weekday'] = self.data['Date'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').weekday(), 1)
        return self

    def drop_columns(self, columns: list):
        """
        删除一列或者多列的数据
        :param columns: 删除的数据列, 如： ['Date']
        :return: self
        """
        self.data.drop(columns, axis=1, inplace=True)
        return self

    def add_increase_col(self):
        """
        添加每日增加的价格，当天Close - Open得到的数值,并保存在新增的数据列Increase中。
        :return: self
        """
        self.data['Increase'] = self.data['Close**'] - self.data['Open*']
        return self

    def normalize(self, columns: list):
        """
        将数据标准化
        :param columns: 要进行标准化的列，如：['Open*', 'High', 'Close**']
        :return:
        """
        for col in columns:
            self.normalizer = MinMaxScaler()
            self.data[col] = self.normalizer.fit_transform(self.data[col].reshape(1, -1)).reshape(-1, 1)
        return self

    def scalar(self, columns: list, scale: float):
        """
        数据放缩，可将选中的列的数据放大和缩小
        :param columns: 要进行scalar的列，如：['Open*', 'High', 'Close**']
        :param scale: 放缩比例，小于1表示缩小，大于1表示放大
        :return: self
        """
        for col in columns:
            self.data[col] = self.data[col] * scale

        return self

    def split(self, train_size: float = 0.6):
        """
        分出训练集和测试集，保存在self.train和self.test中
        :param train_size: 训练集占总数的比例，默认0.6
        :return: self
        """
        split_index = np.floor(len(self.data) * train_size).astype(int)
        print('training size:', split_index)
        print('test size:', len(self.data) - split_index)
        self.train = self.data[:split_index]
        self.test = self.data[split_index:]
        return self

    def generate_train_test_data(self, x_columns: list, y_columns: list, window_len=7, flatten=False):
        """
        产生训练数据和测试数据，默认用最近7天的数据来预测明天的价格
        :param x_columns: 特征列，如：['Open*', 'High', 'Close**', 'Weekday']
        :param y_columns: 预测结果列，如：['Close**']
        :param window_len: 数据窗口长度，默认为7
        :param flatten: 是否返回Flatten后的数据
        :return: (train_x, train_y, test_x, test_y)
        """

        train_x = list()
        train_y = list()
        test_x = list()
        test_y = list()
        if self.train is None or self.test is None:
            self.split()
        # 每一个数据为[window_len, features]
        for i in range(len(self.train) - window_len - 1):
            train_x.append(np.array(self.train[i:(i + window_len)][x_columns]))
            train_y.append(np.array(self.train[i + window_len + 1:i + window_len + 2][y_columns])[0])

        for i in range(len(self.test) - window_len - 1):
            test_x.append(np.array(self.test[i:(i + window_len)][x_columns]))
            test_y.append(np.array(self.test[i + window_len + 1:i + window_len + 2][y_columns])[0])

        if flatten:
            return np.array(train_x).astype(np.float32).reshape((len(train_x), -1)), np.array(train_y).astype(
                np.float32), np.array(test_x).astype(
                np.float32).reshape((len(test_x), -1)), np.array(test_y).astype(np.float32)
        else:
            return np.array(train_x).astype(np.float32), np.array(train_y).astype(np.float32), np.array(test_x).astype(
                np.float32), np.array(test_y).astype(np.float32)


if __name__ == '__main__':
    data = HistoryData('../data/bitcoin-20130428-20180611.csv')
    data.date2weekday().add_increase_col().split()
    # print(data.data)
