import os
import pandas as pd

from sklearn.utils import shuffle

from src.feature_handle.base_data_info import base_info, base_describe
from src.feature_handle.base_utils import count_cn_words
from src.feature_handle.feature_generate import *
from src.feature_handle.feature_transform import log_transform

"""
实验项目
"""
base_data_path = '/home/disk4/weijinqian/baidu/personal-code/comsumer-interests/data/consumer_profile/baidu_app'
data_path = base_data_path + '/baiduapp_comments_greater_20_part_11th_socre'
save_mid_path = base_data_path + '/saved_comments_words_1.txt'
save_final_path = base_data_path + '/saved_comments_words_final.txt'
good_user_id = base_data_path + '/good_user_normal_id.txt'
good_user_to_see_log = base_data_path + '/good_user_to_see_log.txt'
good_user_to_see = base_data_path + '/good_user_to_see.txt'
bad_user_to_see = base_data_path + '/bad_user_to_see.txt'
save_log_path = base_data_path + '/saved_comments_words_log.txt'
output_view_path = base_data_path + '/output_to_view.txt'
good_sample_path = base_data_path + '/good_sample.txt'
bad_sample_path = base_data_path + '/bad_sample.txt'
predict_sample_path = base_data_path + '/predict_data.txt'


def get_agg_feature_by_uid():
    """
    uid作为key，获取数据的分类特征
    :return:
    """
    data = pd.read_csv(data_path, sep='\t',
                       encoding='utf-8', error_bad_lines=False)
    data.columns = ['uid', 'comment_id', 'time',
                    'content_type', 'content', 'comment_score']
    data = data.dropna()
    # 统计文字个数
    data['cn_count'] = data['content'].apply(count_cn_words)
    data = data[data['content'] != '\\N']
    data = data[data['content_type'] != '\\N']
    data = data[data['time'] != '\\N']

    print(base_info(data))
    print(base_describe(data))

    def content_join_func(x): return "----".join([str(i) for i in x])

    # 看字数的分布
    # paint_dist(data, ['cn_count'])
    agg_dict = {
        'cn_count': list_join_func,
        'comment_id': list_join_func,
        'content_type': list_join_func,
        'time': list_time_interval_join_func,
        'comment_score': list_join_func,
        'content': content_join_func
    }
    rename_dict = {
        'cn_count': 'comment_count_path',
        'comment_id': 'comment_id_path',
        'content_type': 'content_type_path',
        'time': 'time_path',
        'comment_score': 'comment_score_path',
        'content': 'comment_path'
    }

    print(data.shape)
    # 获取所有不重复的uid
    all_data = pd.DataFrame(data['uid'].drop_duplicates())
    print(all_data.shape)
    all_data = merge_list(all_data, 'uid', data, agg_dict, rename_dict)

    all_data.to_csv(save_mid_path, sep='\t', index=False)


def feature_eda(data: str):
    all_data = pd.read_csv(data, sep='\t', encoding='utf-8')
    print("read finished.")
    # 评论字数统计变量
    all_data = user_cnt(all_data, 'comment_count_path', 'comment_count')
    all_data = user_max(all_data, 'comment_count_path', 'comment_words_max')
    all_data = user_mean(all_data, 'comment_count_path', 'comment_words_mean')
    all_data = user_min(all_data, 'comment_count_path', 'comment_words_min')
    all_data = user_std(all_data, 'comment_count_path', 'comment_words_std')
    print("words count finished.")

    # 评论类型video：pictext比例
    all_data = user_type_ratio(
        all_data, 'content_type_path', 'type_ratio', 'video', 'pictext')
    all_data = user_type_count(
        all_data, 'content_type_path', 'video_count', 'video')
    all_data = user_type_count(
        all_data, 'content_type_path', 'pictext_count', 'pictext')
    all_data = user_type_count(
        all_data, 'content_type_path', 'other_count', 'other')
    print('content_type finished')

    # 时间间隔统计变量
    all_data = user_max(all_data, 'time_path', 'time_interval_max')
    all_data = user_mean(all_data, 'time_path', 'time_interval_mean')
    all_data = user_min(all_data, 'time_path', 'time_interval_min')
    all_data = user_std(all_data, 'time_path', 'time_interval_std')
    print('time_interval_finished')

    # 评论打分统计变量
    all_data = user_max(all_data, 'comment_score_path', 'comment_score_max')
    all_data = user_mean(all_data, 'comment_score_path', 'comment_score_mean')
    all_data = user_min(all_data, 'comment_score_path', 'comment_score_min')
    all_data = user_std(all_data, 'comment_score_path', 'comment_score_std')
    all_data = user_type_count(
        all_data, 'comment_score_path', 'comment_score_2', '2.0')
    all_data = user_type_count(
        all_data, 'comment_score_path', 'comment_score_1', '1.0')
    all_data = user_type_count(
        all_data, 'comment_score_path', 'comment_score_0', '0.0')
    all_data['percentage_2'] = all_data['comment_score_2'] / \
                               all_data['comment_count']
    all_data['percentage_1'] = all_data['comment_score_1'] / \
                               all_data['comment_count']
    all_data['percentage_0'] = all_data['comment_score_0'] / \
                               all_data['comment_count']
    all_data = all_data.round(3)
    print('comment score is finished.')

    all_data.to_csv(save_final_path, sep='\t', index=False)

    print(base_info(all_data))
    print(base_describe(all_data))

    # print(base_info(comments_word_data))

    # comments_word_data = pd.DataFrame(log_transform(comments_word_data.values),
    #                                   columns=['comment_count', 'comment_words_max', 'comment_count_min',
    #                                            'comment_count_std'])

    # paint_dist(comments_word_data, ['comment_count', 'comment_words_max', 'comment_count_min', 'comment_count_std'])


# 看下数据的信息，内存占用
# 打印阈值卡出去的用户数


def print_num_after_thresold(data: DataFrame, thresolds):
    for key in thresolds:
        print(key + ": " + str(thresolds[key]))
        print(data[data[key] >= thresolds[key]].shape)


def feature_log():
    columns = [
        'uid', 'comment_count', 'comment_words_max',
        'comment_words_mean', 'comment_words_min', 'comment_words_std',
        'type_ratio', 'video_count', 'pictext_count', 'other_count',
        'time_interval_max', 'time_interval_mean', 'time_interval_min',
        'time_interval_std', 'comment_score_max', 'comment_score_mean',
        'comment_score_min', 'comment_score_std', 'commemt_score_2',
        'commemt_score_1', 'percentage_2', 'percentage_1']
    data = pd.read_csv(save_final_path, sep='\t')
    print(base_info(data))
    print(base_describe(data))
    # data = filter_by_threshold_below(data, thresolds)
    data.replace(-1, 0, inplace=True)

    # paint_box(data, columns)
    # 取对数
    log_data = pd.DataFrame(log_transform(data.iloc[:, 1:].values))
    log_data.columns = columns[1:]
    print(type(data[['uid', 'comment_count']]))
    print(log_data.shape)
    print(data.shape)
    data = pd.concat([data[['uid', 'comment_count']], log_data], axis=1)
    data.to_csv(save_log_path, sep='\t', index=False)
    print(base_info(data))
    print(base_describe(data))


def filter_by_threshold_greater(data: DataFrame, thresholds):
    for key in thresholds:
        data = data[data[key] >= thresholds[key]]

    return data


def filter_by_threshold_below(data: DataFrame, thresholds):
    for key in thresholds:
        data = data[data[key] < thresholds[key]]

    return data


def get_model_features():
    return [
        'uid', 'comment_count', 'comment_words_max',
        'comment_words_mean', 'comment_words_min', 'comment_words_std',
        'type_ratio', 'video_count', 'pictext_count', 'other_count',
        'time_interval_max', 'time_interval_mean', 'time_interval_min',
        'time_interval_std', 'comment_score_max', 'comment_score_mean',
        'comment_score_min', 'comment_score_std', 'comment_score_2',
        'comment_score_1', 'comment_score_0', 'percentage_2', 'percentage_1']


def output_view():
    columns = [
        'uid', 'comment_count', 'comment_words_max',
        'comment_words_mean', 'comment_words_min', 'comment_words_std',
        'type_ratio', 'video_count', 'pictext_count', 'other_count',
        'time_interval_max', 'time_interval_mean', 'time_interval_min',
        'time_interval_std', 'comment_score_max', 'comment_score_mean',
        'comment_score_min', 'comment_score_std', 'comment_score_2',
        'comment_score_1', 'comment_score_0', 'percentage_2', 'percentage_1']
    all_data = pd.read_csv(save_final_path, sep='\t', encoding='utf-8')
    data = all_data[columns]
    data.to_csv(output_view_path, sep='\t', index=False)


def filter_duplicate():
    data = pd.read_csv(save_final_path, sep='\t')
    data = data.drop_duplicates(['uid'], keep='last')
    data.to_csv(save_final_path, sep='\t', index=False)


def get_good_users_log():
    data = pd.read_csv(save_log_path, sep='\t')
    origin_data = pd.read_csv(save_mid_path, sep='\t', encoding='utf-8')
    print(base_info(data))
    print(base_describe(data))
    # 对数阈值
    log_thresholds = {'comment_count': 2.5,
                      'comment_words_max': 5,
                      'comment_words_mean': 4,
                      'pictext_count': 1.0
                      }
    # paint_box(data, columns[1:])
    data = filter_by_threshold_greater(data, log_thresholds)
    user_id = data[['uid']]
    user_id = user_id.merge(origin_data, on='uid', how='left')
    user_id = user_id[['uid', 'comment_path']]
    user_id.to_csv(good_user_to_see_log, sep='\t', index=False)


def good_users_normal():
    data = pd.read_csv(save_final_path, sep='\t')
    print(base_info(data))
    print(base_describe(data))
    # 普通阈值
    data = data[data['comment_count'] >= 10]
    data = data[data['comment_words_max'] >= 50]
    data = data[data['comment_words_mean'] >= 40]
    data = data[data['percentage_2'] != 0]
    data = data[data['percentage_0'] <= 0.5]
    return data


def get_good_users_to_see():
    data = good_users_normal()
    user_id = data[['uid', 'comment_path']]
    user_id.to_csv(good_user_to_see, sep='\t', index=False)


def get_good_users_to_model():
    data = good_users_normal()
    print(data.shape)
    user_id = data[get_model_features()]
    user_id['label'] = 1
    user_id.to_csv(good_sample_path, sep='\t', index=False)


def bad_users_normal():
    data = pd.read_csv(save_final_path, sep='\t')
    print(base_info(data))
    print(base_describe(data))
    data = data[data['comment_count'] <= 5]
    data = data[data['comment_words_mean'] <= 20]
    data = data[data['percentage_0'] >= 1]
    return data


def get_bad_users_to_see():
    data = bad_users_normal()
    user_id = data[['uid', 'comment_path']]
    user_id.to_csv(bad_user_to_see, sep='\t', index=False)


def get_bad_users_to_model():
    data = bad_users_normal()
    user_id = data[get_model_features()]
    user_id['label'] = 0
    user_id.to_csv(bad_sample_path, sep='\t', index=False)


def get_predict_users_to_model():
    data = pd.read_csv(save_final_path, sep='\t')
    data = shuffle(data)
    data = data.iloc[:10000]
    data.to_csv(predict_sample_path, sep='\t', index=False)


if __name__ == "__main__":
    # 聚合特征
    # get_agg_feature_by_uid()
    # 统计特征计算起来
    # feature_eda(save_mid_path)
    # 对feature取log
    # feature_log()
    # 用于评估规则效果
    get_good_users_to_see()
    # get_bad_users_to_see()
    # 用于建模
    # get_good_users_to_model()
    # get_bad_users_to_model()
    # get_predict_users_to_model()
    # output_view()
    # filter_duplicate()
    pass
