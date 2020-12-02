# 1. 评论时间间隔，最大值，最小值，平均值
# 2. 评论的标点符号数
import pandas as pd

from src.feature_handle.base_data_info import *
from src.feature_handle.base_utils import reduce_mem_usage
from src.feature_handle.feature_transform import nan_handle, log_transform
from src.feature_handle.view_utils import *

"""

"""

data_path = '/Users/weijinqian/Documents/feature/feature_eda/data/project/saved_comments_words_final.txt'
save_final_path = '/Users/weijinqian/Documents/feature/feature_eda/data/project/saved_comments_words_final_part.txt'
columns = [
    'comment_count', 'comment_words_max',
    'comment_words_mean', 'comment_words_min', 'comment_words_std',
    'type_ratio', 'video_count', 'pictext_count', 'other_count',
    'time_interval_max', 'time_interval_mean', 'time_interval_min',
    'time_interval_std', 'comment_score_max', 'comment_score_mean',
    'comment_score_min', 'comment_score_std', 'commemt_score_2',
    'commemt_score_1', 'percentage_2', 'percentage_1']


def f1():
    data = pd.read_csv(data_path, sep='\t', encoding='utf-8')
    print(data.columns)
    # 降低内存占用
    # data = reduce_mem_usage(data)

    data = data[columns]
    print(data.columns)
    # 做一下数据过滤的操作
    data = pd.DataFrame(nan_handle(data))
    data.columns = columns
    data.to_csv(save_final_path, sep='\t', index=False)


# 打印阈值卡出去的用户数
def print_num_after_thresold(data: DataFrame, thresolds):
    for key in thresolds:
        print(key + ": " + str(thresolds[key]), end='')
        print(data[data[key] > thresolds[key]].shape)


def filter_by_thresold(data: DataFrame, thresolds):
    for key in thresolds:
        data = data[data[key] <= thresolds[key]]

    return data


def f2():
    data = pd.read_csv(save_final_path, sep='\t')
    print(base_info(data))
    print(base_describe(data))
    thresolds = {'comment_count': 25,
                 'comment_words_max': 500,
                 'comment_words_mean': 100,
                 'comment_words_min': 100,
                 'comment_words_std': 100,
                 'type_ratio': 50,
                 'video_count': 50,
                 'pictext_count': 50,
                 'other_count': 50,
                 'time_interval_max': 14 * 86400,
                 'time_interval_mean': 14 * 86400,
                 'time_interval_min': 14 * 86400,
                 'time_interval_std': 14 * 86400,
                 'comment_score_max': 2,
                 'comment_score_mean': 2,
                 'comment_score_min': 2,
                 'comment_score_std': 2,
                 'commemt_score_2': 50,
                 'commemt_score_1': 50,
                 'percentage_2': 1,
                 'percentage_1': 1}
    print_num_after_thresold(data, thresolds)
    # data = filter_by_thresold(data, thresolds)
    data.replace(-1, 0, inplace=True)

    paint_box(data, columns)
    # 取对数
    # data = pd.DataFrame(log_transform(data.values))
    # data.columns = columns
    # columns2 = ['comment_count', 'comment_words_max',
    #             'comment_words_mean', 'comment_words_min', 'comment_words_std',
    #             'type_ratio', 'video_count', 'pictext_count', 'other_count',
    #             'time_interval_max', 'time_interval_mean', 'time_interval_min',
    #             'time_interval_std', 'comment_score_max', 'comment_score_mean',
    #             'comment_score_std', 'percentage_2', 'percentage_1']
    # paint_dist(data, columns2)


f2()
