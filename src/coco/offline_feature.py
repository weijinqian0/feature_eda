import datetime
from abc import ABCMeta, abstractmethod

import pandas as pd

from src.feature_handle.base_utils import to_csv


class OfflineFeatureHandler(metaclass=ABCMeta):
    """
    离线特征处理，某些特征生成的操作，需要加入缓存，保存文件中，便于下次使用
    使用时直接扩展相关的方法
    """

    def __init__(self, data_path, saved_path, columns):
        self.data = pd.read_csv(data_path, sep='\t')
        self.data.columns = columns
        self.saved_path = saved_path

    def fill_nan(self):
        return self.data, False

    def feature_filter(self):
        return self.data, False

    def feature_select(self):
        return self.data, False

    def feature_generate(self):
        return self.data, False

    def save_data(self, name):
        ver = datetime.datetime.now().strftime('%Y-%m-%d%H:%M:%S')
        path = self.saved_path + name + str(ver) + '.csv'
        to_csv(self.data, path)

    def pipeline(self):
        self.data, is_save = self.fill_nan()
        if is_save:
            self.save_data('fill_nan')
        self.data, is_save = self.feature_generate()
        if is_save:
            self.save_data('feature_generate')
        self.data, is_save = self.feature_filter()
        if is_save:
            self.save_data('feature_filter')
        self.data, is_save = self.feature_select()
        if is_save:
            self.save_data('feature_select')


if __name__ == "__main__":
    data_path = ''
    save_path = ''
    # 离线特征处理并保存
    OfflineFeatureHandler(data_path, save_path).pipeline()
