import datetime

from pandas import DataFrame
import pandas as pd

from src.feature_handle.base_data_info import base_info, base_describe
from src.feature_handle.base_utils import to_csv
from src.feature_handle.feature_embedded import lgb_embeded_1

"""
真实情况下，feature 是不断迭代的
每次迭代的时候，需要保存当前的数据副本或特征的副本或代码的副本

"""


class OnlineFeatureHandler():
    """
    在线特征处理，在调整模型的时候，需要调整某些特征的处理，或者临时需要添加某些特征
    """

    def __init__(self, data_path, label_name):
        self.data = pd.read_csv(data_path, sep='\t')
        self.label_name = label_name

    def fill_nan(self):
        return self.data

    def feature_filter(self):
        return self.data

    def feature_select(self):
        return self.data

    def feature_generate(self):
        return self.data

    def pipeline(self):
        self.data = self.fill_nan()
        self.data = self.feature_generate()
        self.data = self.feature_filter()
        self.data = self.feature_select()


class OfflineFeatureHandler(object):
    """
    离线特征处理，某些特征生成的操作，需要加入缓存，保存文件中，便于下次使用
    使用时直接扩展相关的方法
    """

    def __init__(self, data_path, saved_path):
        self.data = pd.read_csv(data_path, sep='\t')
        self.data_X = None
        self.data_y = None
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


def fill_nan(train_data: DataFrame, label_name):
    """
    缺失值处理
    :param train_data:
    :param label_name:
    :return:
    """
    train_target = train_data[label_name]
    train_data = train_data.drop(label_name, axis=1)
    train_data.fillna(value=train_data.median())

    return train_data, train_target


def feature_select(name, train_data: DataFrame, target, feature_name):
    if name == "lgb":
        return lgb_embeded_1(train_data, target, feature_name, '0.1*mean')
    else:
        return train_data, target, feature_name, []


def feature_preprocessor(data_train, label_name, select_fun=''):
    """
    数据预处理
    :type select_fun: str 特征选择的方法
    :return:
    """

    print(base_info(data_train))
    print(base_describe(data_train))

    # 填充nan
    train_data, train_target = fill_nan(data_train, label_name)

    X = train_data.values
    y = train_target.values
    # 特征选择
    X, feature_score_dict_sorted, feature_used_name, feature_not_used_name = feature_select(
        select_fun, X, y, train_data.columns.tolist())

    print("当前选中的特征维度" + str(len(feature_used_name)))
    print(feature_used_name)
    print(feature_not_used_name)
    return X, y, feature_used_name


if __name__ == "__main__":
    data_path = ''
    save_path = ''
    # 离线特征处理并保存
    OfflineFeatureHandler(data_path, save_path).pipeline()
