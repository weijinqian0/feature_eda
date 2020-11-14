import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

import seaborn as sns
from scipy import stats

import warnings

from src.base_data_info import *
from src.view_utils import *

warnings.filterwarnings("ignore")

"""
数据基础信息文件
"""

train_data_file = '../data/zhengqi_train.txt'
test_data_file = '../data/zhengqi_test.txt'
train_data = pd.read_csv(train_data_file, sep='\t', encoding='utf-8')
test_data = pd.read_csv(test_data_file, sep='\t', encoding='utf-8')

# 得到基本信息
# print(base_info(train_data))
# print(base_describe(train_data))
#
# 开始绘制箱型图
# print(train_data.columns)
# paint_box(train_data, train_data.columns)

# from sklearn.linear_model import Ridge

# X_train = train_data.iloc[:, 0:-1]
# y_train = train_data.iloc[:, -1]
# outliers = find_outliers(Ridge(), X_train, y_train)

# paint_dist(train_data, train_data.columns[:2])

paint_heatmap(train_data, ['V5', 'V9', 'V11', 'V17', 'V22', 'V28'])

if __name__ == '__main__':
    pass
