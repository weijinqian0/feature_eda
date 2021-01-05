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

    @abstractmethod
    def fill_nan(self):
        pass

    @abstractmethod
    def feature_filter(self):
        pass

    @abstractmethod
    def feature_select(self):
        pass

    @abstractmethod
    def feature_generate(self):
        pass

    def save_data(self, data, name):
        ver = datetime.datetime.now().strftime('%Y-%m-%d%H:%M:%S')
        path = self.saved_path + name + str(ver) + '.csv'
        to_csv(data, path)
        return path

    def pipeline(self):
        self.fill_nan()
        self.feature_generate()
        self.feature_filter()
        self.feature_select()


if __name__ == "__main__":
    data_path = ''
    save_path = ''
    columns = []
    # 离线特征处理并保存
    OfflineFeatureHandler(data_path, save_path, columns).pipeline()
