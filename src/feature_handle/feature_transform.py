import matplotlib
from pandas import DataFrame
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, Binarizer, OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd

matplotlib.use('TkAgg')
from scipy import stats

import warnings

warnings.filterwarnings("ignore")

"""
特征转换文件，这里都是对特征做各种基本处理
"""


def scale_minmax(col):
    return (col - col.min()) / (col.max() - col.min())


def standard_scale(data: DataFrame):
    """
    标准化, x'=x-mean(x)/S S为方差
    由于fit_transform 之后，得到的是一个numpy array，所以需要使用pandas转换回来
    :param data:
    :return:
    """
    return pd.DataFrame(StandardScaler().fit_transform(data), columns=data.columns, index=data.index)


def minmax_scale(data):
    """
    区间缩放法 x'=x-Min/Max-Min
    :param data:
    :return:
    """
    return pd.DataFrame(MinMaxScaler().fit_transform(data), columns=data.columns, index=data.index)


def normalizer(data):
    """
    归一化 x'=x/sqrt(sum(x[j]^2))
    :param data:
    :return:
    """
    return pd.DataFrame(Normalizer().fit_transform(data), columns=data.columns, index=data.index)


def binarizer(data, threshold):
    """
    讲数据二值化
    :param data:
    :param threshold:
    :return:
    """
    return pd.DataFrame(Binarizer(threshold=threshold).fit_transform(data), columns=data.columns, index=data.index)


def one_hot_encode(data, column):
    """
    对数据进行one_hot编码
    :param data:
    :param column:
    :return:
    """
    return pd.DataFrame(OneHotEncoder(categories='auto').fit_transform(data[column]), columns=data.columns,
                        index=data.index)


def nan_handle(data):
    """
    缺失值处理，这里默认是平均值填充，但是可以自定义
    :return:
    """
    return pd.DataFrame(SimpleImputer().fit_transform(data), columns=data.columns, index=data.index)


def poly_transform(data):
    """
    使用多项式进行转换
    :param data:
    :return:
    """
    return pd.DataFrame(PolynomialFeatures().fit_transform(data), columns=data.columns, index=data.index)


def ordinal_transform(data, column: str):
    """
    将文本信息转换为类别数字信息
    :param column: 某一列
    :param data:
    :return:
    """
    return pd.DataFrame(OrdinalEncoder().fit_transform(data[[column]]), columns=data.columns, index=data.index)


def column_transform(data, strategy):
    """
    可以添加策略，使用统一的策略添加能力，0.20版本以上使用
    :param data:
    :param strategy:
    :return:
    """
    return ColumnTransformer(strategy, remainder='passthrough').fit_transform(data)


def log_transform(data: np.ndarray):
    """
    对数变化，非常有用，对于分布不均的，类似gama函数，长尾的数据
    :param data: 这里的类型是np.ndarray 千万不要用错
    :return:
    """
    return FunctionTransformer(np.log1p, validate=False).fit_transform(data)


def boxcox(data: DataFrame):
    """
    boxcox使用场景：线性回归要求数据服从正太分布
    解释：对于线性回归模型,当因变量服从正态分布,误差项满足高斯–马尔科夫条件（零均值、等方差、不相关）时,回归参数的最小二乘估计是一致最小方差无偏估计.
    解释二：线性回归是广义线性模型，它的函数指数簇就是高斯分布。
    在对data做boxcox之前，需要对数据先做归一化，归一化之后才能使用box-cox处理，使得数据服从正太分布
    :param data: DataFrame
    :return:
    """
    columns = list(data.columns)
    data[columns] = data[columns].apply(scale_minmax, axis=0)
    for var in list(data.columns):
        trans_var, lambda_var = stats.boxcox(data[var].dropna() + 1)
        trans_var = scale_minmax(trans_var)

    return trans_var
