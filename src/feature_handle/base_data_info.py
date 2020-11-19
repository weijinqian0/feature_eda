import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import warnings

warnings.filterwarnings("ignore")

"""
数据基础信息文件
"""


def base_info(data):
    return data.info()


def base_describe(data):
    return data.describe()


