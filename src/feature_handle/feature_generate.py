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


# reduce memory
def reduce_mem_usage(df: DataFrame, verbose=True):
    """
    判断每一列的数值大小，从而降低存储所需要内存
    :param df:
    :param verbose:
    :return:
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.iinfo(np.float16).min and c_max < np.iinfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.iinfo(np.float32).min and c_max < np.iinfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
    print("Decreased by [:1f]%".format(100 * (start_mem - end_mem) / start_mem))
    return df


list_join_func = lambda x: " ".join([str(i) for i in x])

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


"""
开启统计特征
"""


# 统计特征
def cnt_(x):
    try:
        return len(x.split(' '))
    except:
        return -1


# 定义统计数据唯一值总数
def nunique_(x):
    try:
        return len(set(x.split(' ')))
    except:
        return -1


# 统计数据序列中最大的值
def max_(x):
    try:
        return np.max([float(i) for i in x.split(' ')])
    except:
        return -1


# 统计数据序列中最小的值
def min_(x):
    try:
        return np.min([float(i) for i in x.split(' ')])
    except:
        return -1


# 标准差
def std_(x):
    try:
        return np.std([float(i) for i in x.split(' ')])
    except:
        return -1


# 定义统计数据中topN 数据的函数
def most_n(x, n):
    try:
        return Counter(x.split(' ')).most_common(n)[n - 1][0]
    except:
        return -1


#  定义统计数据中topN数据总数的函数
def most_n_cnt(x, n):
    try:
        return Counter(x.split(' ')).most_common(n)[n - 1][1]
    except:
        return -1


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
    test_data = pd.read_csv('data/demo3_data/data_format1/test_format1.csv')
    train_data = pd.read_csv('data/demo3_data/data_format1/train_format1.csv')

    user_info = pd.read_csv('data/demo3_data/data_format1/user_info_format1.csv')
    user_log = pd.read_csv('data/demo3_data/data_format1/user_log_format1.csv')

    all_data = train_data.append(test_data)
    all_data = all_data.merge(user_info, on=['user_id'], how='left')
    del train_data, test_data, user_info
    gc.collect()

    # 用户行为日志先按时间排好序，然后直接生成路径序列
    user_log = user_log.sort_values(['user_id', 'time_stamp'])
    all_data = merge_list(all_data, 'user_id', user_log, agg_dict, rename_dict)

    all_data_test = all_data.head(2000)

    # 店铺特征统计！！！统计与店铺特点有关的特征，如店铺、商品、品牌等
    # 统计用户点击、浏览、加购、购买行为
    # 总次数
    all_data_test = user_cnt(all_data_test, 'seller_path', 'user_cnt')
    # 不同店铺个数
    all_data_test = user_nunique(all_data_test, 'seller_path', 'seller_nunique')
    # 不同品牌的个数
    all_data_test = user_nunique(all_data_test, 'cat_path', 'cat_nunique')
    # 不同品类的个数
    all_data_test = user_nunique(all_data_test, 'brand_path', 'brand_nunique')
    # 不同商品的个数
    all_data_test = user_nunique(all_data_test, 'item_path', 'item_nunique')
    # 活跃天数
    all_data_test = user_nunique(all_data_test, 'time_stamp_path', 'time_stamp_nunique')
    # 不同用户行为种数
    all_data_test = user_nunique(all_data_test, 'action_type_path', 'action_type_nunique')
    # 最晚时间
    all_data_test = user_max(all_data_test, 'action_type_path', 'time_stamp_max')
    # 最早时间
    all_data_test = user_min(all_data_test, 'action_type_path', 'time_stamp_min')
    # 活跃天数方差
    all_data_test = user_std(all_data_test, 'action_type_path', 'time_stamp_std')
    # 最早与最晚相差天数
    all_data_test['time_stamp_range'] = all_data_test['time_stamp_max'] - all_data_test['time_stamp_min']
    # 用户最喜欢的店铺
    all_data_test = user_most_n(all_data_test, 'seller_path', 'seller_most_1', n=1)
    # 最喜欢的品牌
    all_data_test = user_most_n(all_data_test, 'cat_path', 'cat_most_1', n=1)
    # 最常见的行为动作
    all_data_test = user_most_n(all_data_test, 'brand_path', 'brand_most_1', n=1)
    # 用户最喜欢的店铺  行为次数
    all_data_test = user_most_n_cnt(all_data_test, 'seller_path', 'seller_most_1_cnt', n=1)
    # 用户最喜欢的类目 行为次数
    all_data_test = user_most_n_cnt(all_data_test, 'cat_path', 'cat_most_1_cnt', n=1)
    # 最喜欢的品牌 行为次数
    all_data_test = user_most_n_cnt(all_data_test, 'brand_path', 'brand_most_1_cnt', n=1)
    # 最常见的行为动作 行为次数
    all_data_test = user_most_n_cnt(all_data_test, 'action_type_path', 'action_most_1_cnt', n=1)

    # 用户统计特征：对用户的点击、加购、购买、收藏等特征进行统计
    # 点击次数
    all_data_test = user_col_cnt(all_data_test, ['seller_path'], '0', 'user_cnt_0')
    # 加购次数
    all_data_test = user_col_cnt(all_data_test, ['seller_path'], '1', 'user_cnt_1')

    # ...
    print(all_data_test.columns)

    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    from scipy import sparse

    tfidfVec = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS, ngram_range=(1, 1), max_features=100)
    columns_list = ['seller_path']
    for i, col in enumerate(columns_list):
        tfidfVec.fit(all_data_test[col])
        data_ = tfidfVec.transform(all_data_test[col])
        if i == 0:
            data_cat = data_
        else:
            data_cat = sparse.hstack((data_cat, data_))

    df_tdidf = pd.DataFrame(data_cat.toarray())

    # 特征重命名与特征合并
    df_idf = pd.DataFrame(data_cat.toarray())
    df_tdidf.columns = ['tfidf_' + str(i) for i in df_tdidf.columns]
    all_data_test = pd.concat([all_data_test, df_tdidf], axis=1)

    # 嵌入特征
    import gensim

    model = gensim.models.Word2Vec(all_data_test['seller_path'].apply(lambda x: x.split(' ')),
                                   size=100, window=5,
                                   min_count=5, workers=4)

    model.save('product2vec.model')
    model = gensim.models.Word2Vec.load("product2vec.model")

    df_embedding = get_mean_w2v(all_data_test, 'seller_path', model, 100)
    df_embedding.columns = ['embedding_' + str(i) for i in df_embedding.columns]

    all_data_test = pd.concat([all_data_test, df_embedding], axis=1)

