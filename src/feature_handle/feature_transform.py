import matplotlib

from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, Binarizer, OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer
from numpy import log1p
from sklearn.impute import SimpleImputer

matplotlib.use('TkAgg')
from scipy import stats

import warnings

warnings.filterwarnings("ignore")

"""
特征转换文件，这里都是对特征做各种基本处理
"""


def scale_minmax(col):
    return (col - col.min()) / (col.max() - col.min())


def standard_scale(data):
    """
    标准化, x'=x-mean(x)/S S为方差
    :param data:
    :return:
    """
    return StandardScaler().fit_transform(data)


def minmax_scale(data):
    """
    区间缩放法 x'=x-Min/Max-Min
    :param data:
    :return:
    """
    return MinMaxScaler().fit_transform(data)


def normalizer(data):
    """
    归一化 x'=x/sqrt(sum(x[j]^2))
    :param data:
    :return:
    """
    return Normalizer().fit_transform(data)


def binarizer(data, threshold):
    """
    讲数据二值化
    :param data:
    :param threshold:
    :return:
    """
    return Binarizer(threshold=threshold).fit_transform(data)


def one_hot_encode(data, column):
    """
    对数据进行one_hot编码
    :param data:
    :param column:
    :return:
    """
    return OneHotEncoder(categories='auto').fit_transform(data[column])


def nan_handle(data):
    """
    缺失值处理，这里默认是平均值填充，但是可以自定义
    :return:
    """
    return SimpleImputer().fit_transform(data)


def poly_transform(data):
    """
    使用多项式进行转换
    :param data:
    :return:
    """
    return PolynomialFeatures().fit_transform(data)


def log_transform(data):
    """
    对数变化，非常有用，对于分布不均的，类似gama函数，长尾的数据
    :param data:
    :return:
    """
    return FunctionTransformer(log1p, validate=False).fit_transform(data)


def boxcox(data):
    """
    boxcox使用场景：线性回归要求数据服从正太分布
    解释：对于线性回归模型,当因变量服从正态分布,误差项满足高斯–马尔科夫条件（零均值、等方差、不相关）时,回归参数的最小二乘估计是一致最小方差无偏估计.
    解释二：线性回归是广义线性模型，它的函数指数簇就是高斯分布。
    在对data做boxcox之前，需要对数据先做归一化，归一化之后才能使用box-cox处理，使得数据服从正太分布
    :param data: DataFrame
    :return:
    """
    columns = list(data.columns)
    data[columns] = data[columns].apply(scale_minmax(), axis=0)
    for var in list(data.columns):
        trans_var, lambda_var = stats.boxcox(data[var].dropna() + 1)
        trans_var = scale_minmax(trans_var)

    return trans_var
