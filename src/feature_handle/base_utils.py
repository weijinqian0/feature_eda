"""
特征处理过程中常用的函数
"""
from pandas import DataFrame
import numpy as np


def count_cn_words(data):
    """
    获取中文字个数
    """
    if not isinstance(data, str):
        return 0
    count = 0
    for s in data:
        # 中文字符范围
        if u'\u4e00' <= s <= u'\u9fff':
            count += 1
    return count


# 降低内存存储大小的
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
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
    print("Decreased by {:1f}%".format(100 * (start_mem - end_mem) / start_mem))
    return df


def columns_drop(data, columns_to_drop):
    """
    删除部分列
    :param data:
    :param columns_to_drop:
    :return:
    """
    for col in columns_to_drop:
        data.drop(col, axis=1, inplace=True)


def to_csv(data: DataFrame, path):
    """
    按照csv格式保存到文件中，并去掉index
    :param data:
    :param path:
    :return:
    """
    data.to_csv(path, sep='\t', index=False)


if __name__ == "__main__":
    print(count_cn_words('😣'))
