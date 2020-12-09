"""
主要讲构造特征的几种方式，先简单粗糙的给出代码，之后再重构代码

先扩展特征，再进行特征过滤、嵌入、包装等方法进行处理
统计特征、组合特征、TF_IDF、嵌入特征、Stacking 特征
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame
from scipy import stats

import gc
from collections import Counter
import copy

import warnings

warnings.filterwarnings('ignore')

list_join_func = lambda x: " ".join([str(i) for i in x])


def list_time_interval_join_func(x):
    times = sorted([int(i) for i in x if str(i).isdigit()])
    cur_time = ''
    intervals = []
    for idx, time in enumerate(times):
        if idx == 0:
            cur_time = time
        else:
            intervals.append(int(time) - int(cur_time))

    return " ".join([str(i) for i in intervals])


# 将需要聚合groupby处理的列写成dict
agg_dict = {
    'item_id': list_join_func,
    'cat_id': list_join_func,
    'seller_id': list_join_func,
    'brand_id': list_join_func,
    'time_stamp': list_join_func,
    'action_type': list_join_func
}

rename_dict = {
    'item_id': 'item_path',
    'cat_id': 'cat_path',
    'seller_id': 'seller_path',
    'brand_id': 'brand_path',
    'time_stamp': 'time_stamp_path',
    'action_type': 'action_type_path'
}


def merge_list(df_ID, join_columns, df_data: DataFrame, agg_dict, rename_dict):
    """
    这个函数是把每个id相关的行都列出来，这样可以根据id学习出一个word2vec
    :param df_ID: 目标数据
    :param join_columns: 聚合指定数组，可以传入数组，也可以传入str
    :param df_data: 序列型的数据源
    :param agg_dict: 数据源根据对应的id来进行分组，分组之后的数据使用这里的value进行聚合
    :param rename_dict: 聚合之后的数据命名
    :return: 将生产出的特征融合到原始数据中
    """
    df_data = df_data.groupby(join_columns).agg(agg_dict).reset_index().rename(columns=rename_dict)
    df_ID = df_ID.merge(df_data, on=join_columns, how='left')
    return df_ID


def add_agg_feature_names(df: DataFrame, df_group: DataFrame, group_cols, value_col, agg_ops, col_names):
    """
    统计特征处理函数
    :param df: 添加特征的dataframe
    :param df_group: 特征生成数的数据集
    :param group_cols: group by的列
    :param value_col: 被统计的列
    :param agg_ops:处理方式，包括count,mean,sum,std,max,min,nunique
    :param col_names:新特征的名称
    :return:
    """
    df_group[value_col] = df_group[value_col].astype('float')
    df_agg = pd.DataFrame(df_group.groupby(group_cols)[value_col].agg(agg_ops)).reset_index()

    df_agg.columns = group_cols + col_names
    df = df.merge(df_agg, on=group_cols, how='left')
    return df


def add_agg_feature(df: DataFrame, df_group: DataFrame, group_cols, value_col, agg_ops, keyword):
    col_names = []
    for op in agg_ops:
        col_names.append(keyword + '_' + value_col + '_' + op)
    df = add_agg_feature_names(df, df_group, group_cols, value_col, agg_ops, col_names)
    return df


def add_count_new_feature(df: DataFrame, df_group: DataFrame, group_cols, new_feature_name):
    """
    因为count的函数比较多，因此开发专门提取count特征的函数
    :param df:
    :param df_group:
    :param group_cols:
    :param new_feature_name:
    :return:
    """
    df_group[new_feature_name] = 1
    df_group = df_group.groupby(group_cols).agg('sum').reset_index()
    df = df.merge(df_group, on=group_cols, how='left')
    return df


"""
开启统计特征
"""


# 统计特征
def cnt_(x):
    try:
        return len(x.split(' '))
    except:
        return 0


# 定义统计数据唯一值总数
def nunique_(x):
    try:
        return len(set(x.split(' ')))
    except:
        return 0


# 统计数据序列中最大的值
def max_(x):
    try:
        return np.max([float(i) for i in x.split(' ')])
    except:
        return 0


# 统计数据序列中最小的值
def min_(x):
    try:
        return np.min([float(i) for i in x.split(' ')])
    except:
        return 0


def mean_(x):
    try:
        return np.mean([float(i) for i in x.split(' ')])
    except:
        return 0


# 标准差
def std_(x):
    try:
        return np.std([float(i) for i in x.split(' ')])
    except:
        return 0


# 定义统计数据中topN 数据的函数
def most_n(x, n):
    try:
        return Counter(x.split(' ')).most_common(n)[n - 1][0]
    except:
        return 0


#  定义统计数据中topN数据总数的函数
def most_n_cnt(x, n):
    try:
        return Counter(x.split(' ')).most_common(n)[n - 1][1]
    except:
        return 0


def type_ratio(x, type1, type2):
    try:
        c = Counter(x.split(' '))
        return round(c[type1] / (c[type2] + 1), 2)
    except:
        return 0


def type_count(x, type1):
    try:
        c = Counter(x.split(' '))
        return c[type1]
    except:
        return 0


def user_cnt(df_data, single_col, name):
    df_data[name] = df_data[single_col].apply(cnt_)
    return df_data


def user_nunique(df_data, single_col, name):
    df_data[name] = df_data[single_col].apply(nunique_)
    return df_data


def user_max(df_data, single_col, name):
    df_data[name] = df_data[single_col].apply(max_)
    return df_data


def user_min(df_data, single_col, name):
    df_data[name] = df_data[single_col].apply(min_)
    return df_data


def user_mean(df_data, single_col, name):
    df_data[name] = df_data[single_col].apply(mean_)
    return df_data


def user_std(df_data, single_col, name):
    df_data[name] = df_data[single_col].apply(std_)
    return df_data


def user_most_n(df_data, single_col, name, n=1):
    func = lambda x: most_n(x, n)
    df_data[name] = df_data[single_col].apply(func)
    return df_data


def user_most_n_cnt(df_data, single_col, name, n=1):
    func = lambda x: most_n_cnt(x, n)
    df_data[name] = df_data[single_col].apply(func)
    return df_data


def user_type_ratio(df_data, single_col, name, type1, type2):
    def func(x): return type_ratio(x, type1, type2)

    df_data[name] = df_data[single_col].apply(func)
    return df_data


def user_type_count(df_data, single_col, name, type1):
    def func(x): return type_count(x, type1)

    df_data[name] = df_data[single_col].apply(func)
    return df_data


def col_cnt_(df_data, columns_list, action_type):
    try:
        data_dict = {}
        col_list = copy.deepcopy(columns_list)
        if action_type != None:
            col_list += ['action_type_path']

        for col in col_list:
            data_dict[col] = df_data[col].split(' ')

        path_len = len(data_dict)

        data_out = []
        for i_ in range(path_len):
            data_txt = ''
            for col_ in columns_list:
                if data_dict['action_type_path'][i_] == action_type:
                    data_txt += '_' + data_dict[col_][i_]

            data_out.append(data_txt)

        return len(data_out)

    except:
        return -1


def col_nunique_(df_data, columns_list, action_type):
    try:
        data_dict = {}
        col_list = copy.deepcopy(columns_list)
        if action_type != None:
            col_list += ['action_type_path']

        for col in col_list:
            data_dict[col] = df_data[col].split(' ')

        path_len = len(data_dict)

        data_out = []
        for i_ in range(path_len):
            data_txt = ''
            for col_ in columns_list:
                if data_dict['action_type_path'][i_] == action_type:
                    data_txt += '_' + data_dict[col_][i_]

            data_out.append(data_txt)

        return len(set(data_out))

    except:
        return -1


def user_col_cnt(df_data: DataFrame, columns_list, action_type, name):
    df_data[name] = df_data.apply(lambda x: col_cnt_(x, columns_list, action_type), axis=1)
    return df_data


def user_col_nunique(df_data: DataFrame, columns_list, action_type, name):
    df_data[name] = df_data.apply(lambda x: col_nunique_(x, columns_list, action_type), axis=1)
    return df_data


def mean_w2v_(x, model, size=100):
    try:
        i = 0
        for word in x.split(' '):
            if word in model.wv.vocab:
                i += 1
                if i == 1:
                    vec = np.zeros(size)
                vec += model.wv[word]

        return vec / i
    except:
        return np.zeros(size)


def get_mean_w2v(df_data: DataFrame, columns, model, size):
    data_array = []
    for index, row in df_data.iterrows():
        w2v = mean_w2v_(row[columns], model, size)
        data_array.append(w2v)

    return pd.DataFrame(data_array)


if __name__ == "__main__":
    pass
