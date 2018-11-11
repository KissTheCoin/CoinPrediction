#!/usr/bin/python
# coding: utf-8
# Author: Ross
import pandas as pd
import datetime
import numpy as np
import time
import warnings
import platform
import os
import logging

warnings.simplefilter('always', category=DeprecationWarning)
with warnings.catch_warnings():
    if int(platform.python_version_tuple()[0]) >= 3:
        warnings.warn("This script is not compatible with python3 or over", DeprecationWarning)


class HistoryData:
    API = 'https://coinmarketcap.com/currencies/{symbol}/historical-data/?start={start_time}&end={end_time}'

    def __init__(self, symbol, start_time='20130428', end_time=time.strftime('%Y%m%d')):
        self.symbol = symbol
        self.data = pd.DataFrame()
        self.start_time = start_time
        self.end_time = end_time

    def get_data(self, start_time=None, end_time=None):
        """

        :param symbol: 币种
        :param start_time: 数据开始时间
        :param end_time: 数据结束时间
        :return:
        """
        # 默认为网站记录的最早时间
        if start_time is not None:
            self.start_time = start_time
        if end_time is not None:
            self.end_time = end_time

        url = self.API.format(symbol=self.symbol, start_time=self.start_time, end_time=self.end_time)
        coin_info = pd.read_html(url)[0]
        coin_info = coin_info.assign(Date=pd.to_datetime(coin_info['Date']))
        try:
            coin_info.loc[coin_info['Volume'] == '-', 'Volume'] = 0
        except TypeError:
            pass
        coin_info['Volume'] = coin_info['Volume'].astype('int64')
        self.data = coin_info[::-1]

    def to_csv(self, path=None):
        '''

        :param path: CSV保存路径
        :return:
        '''
        if path is None:
            path = os.path.join('..', 'data', '-'.join([self.symbol, self.start_time, self.end_time]) + '.csv')

        path = os.path.abspath(path)

        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        self.data.to_csv(path, index=False)
        logging.info('csv file saved to ' + path)


if __name__ == '__main__':
    test = HistoryData('eos')
    test.get_data()
    test.to_csv()
