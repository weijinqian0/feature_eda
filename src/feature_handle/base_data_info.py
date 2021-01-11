import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame
from scipy import stats

import warnings

warnings.filterwarnings("ignore")

"""
数据基础信息文件
"""


def base_info(data):
    with pd.option_context('display.max_columns', 2000):
        print(data.info())


def base_head(data):
    with pd.option_context('display.max_columns', 2000):
        print(data.head())


def base_describe(data):
    with pd.option_context('display.max_columns', 2000):
        print(data.describe(include='all'))


def base_isnull(data):
    return data.isnull().sum()


def base_border_explore(data: DataFrame, column):
    print("data " + column)
    print(data[~data[column].isna()][column].min())
    print(data[~data[column].isna()][column].max())

    print(data[~data[column].isna()][column].count())
    print(data[~data[column].isna()][column].sum())

    print(data[~data[column].isna()][column].value_counts())

