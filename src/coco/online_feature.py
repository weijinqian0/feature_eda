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


class OnlineFeatureHandler(object):
    """
    在线特征处理，在调整模型的时候，需要调整某些特征的处理，或者临时需要添加某些特征
    承担了切分的功能
    """

    def __init__(self, label_name, data=None, data_path=None):
        if data is None:
            if data_path is not None:
                self.data = pd.read_csv(data_path, sep='\t')
            else:
                raise BaseException('Data path is none.')
        else:
            self.data = data
        self.label_name = label_name
        self.data_y = self.data[label_name]
        self.data_X = self.data.drop(label_name, axis=1)
        self.columns = self.data_X.columns.tolist()
        self.feature_used_name = []

    def info(self):
        base_info(self.data)
        base_describe(self.data)

    def fill_nan(self):
        return self.data_X.fillna(self.data_X.median())

    def feature_filter(self):
        return self.data_X

    def feature_select(self):
        X = self.data_X.values
        y = self.data_y.values
        X, feature_score_dict_sorted, feature_used_name, feature_not_used_name \
            = feature_select('', X, y, self.columns)
        data_X = pd.DataFrame(X, columns=feature_used_name)

        self.feature_used_name = feature_used_name
        print("当前选中的特征维度" + str(len(feature_used_name)))
        print(feature_used_name)
        print(feature_not_used_name)

        return data_X

    def feature_generate(self):
        return self.data_X

    def pipeline(self):
        self.info()
        self.data_X = self.fill_nan()
        self.data_X = self.feature_generate()
        self.data_X = self.feature_filter()
        self.data_X = self.feature_select()

    def output(self):
        return self.data_X, self.data_y, self.feature_used_name


def feature_select(name, train_data: DataFrame, target, feature_name):
    if name == "lgb":
        return lgb_embeded_1(train_data, target, feature_name, '0.1*mean')
    else:
        return train_data, target, feature_name, []
