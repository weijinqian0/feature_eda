import matplotlib
from array import array

matplotlib.use('TkAgg')

import warnings

warnings.filterwarnings("ignore")

from sklearn.feature_selection import VarianceThreshold, SelectKBest
from scipy.stats import pearsonr
from sklearn.feature_selection import chi2
from minepy import MINE

"""
特征过滤的方法
基本思想是：基于方差、相关性、卡方、皮尔逊系数、互信息
将与label无关的特征过滤出去
"""


def variance_filter(data):
    """
    使用方差选择法，
    :param data:
    :return:
    """
    return VarianceThreshold(threshold=3).fit_transform(data)


def pearsonr_filter(data, target):
    """
    使用皮尔逊系数进行过滤
    :param data:
    :param target: label值
    :return:
    """
    return SelectKBest(lambda X, Y: array(map(lambda x: pearsonr(x, Y), X.T)).T, k=2).fit_transform(data,
                                                                                                    data[target])


def chi2_filter(data, target):
    """
    使用卡方过滤的方式
    :param data:
    :param target:
    :return:
    """
    return SelectKBest(chi2, k=2).fit_transform(data, data[target])


# 由于MINE的设计不是函数式的，定义mic方法将其为函数式的，返回一个二元组，二元组的第2项设置成固定的P值0.5
def mic(x, y):
    m = MINE()
    m.compute_score(x, y)
    return m.mic(), 0.5


def mutual_info_filter(data, target):
    """
    使用互信息过滤
    :param data:
    :return:
    """
    return SelectKBest(lambda X, Y: array(map(lambda x: mic(x, Y), X.T)).T, k=2).fit_transform(data, data[target])


