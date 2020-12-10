from itertools import cycle

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.utils import shuffle
from sklearn.utils.multiclass import type_of_target
import pickle
import time
import datetime

from src.feature_handle.base_data_info import base_info, base_describe
from src.feature_handle.base_utils import columns_drop, to_csv
from src.feature_handle.feature_embedded import lgb_embeded_1
from src.model.statistic_model.classfier_model import LrClassifier


def get_model_features():
    return ['uid', 'comment_count', 'comment_words_max', 'comment_words_mean',
            'comment_words_min', 'comment_words_std', 'type_ratio', 'video_count',
            'pictext_count', 'other_count', 'time_interval_max',
            'time_interval_mean', 'time_interval_min', 'time_interval_std',
            'comment_score_max', 'comment_score_mean', 'comment_score_min',
            'comment_score_std', 'comment_score_2', 'comment_score_1',
            'comment_score_0', 'percentage_2', 'percentage_1', 'percentage_0',
            'comment_score_2_cross', 'comment_score_1_cross',
            'comment_score_0_cross']


def feature_select(name, train_data, target, feature_name):
    if name == "lgb":
        return lgb_embeded_1(train_data, target, feature_name, '0.1*mean')
    else:
        return train_data, target, feature_name, []


def lr_model(train_data, train_target, test_data, test_target):
    clf = LogisticRegression(random_state=0, solver='lbfgs',
                             multi_class='multinomial').fit(train_data, train_target)
    clf.score(test_data, test_target)


def classifier_model(model_name):
    if model_name == 'lr':
        return LrClassifier()


def train_model(data_train, data_predict, result_path, model_path, threshold=0.5, label='label',
                is_save_model=True):
    """
    提供一键调用方式
    :param result_path: 结果保存位置
    :param data_train:
    :param data_predict:
    :return:
    """

    X, y, X_test, len_feature_choose, data_predict_user_id = feature_preprocessor(data_train, data_predict, label)
    # 选择一个模型
    model_name = 'lr'
    model = classifier_model(model_name)

    model.fit(X, y)

    if is_save_model:
        model.save_model(model_path)

    predict_label, predict_proba = model.predict(X_test)

    # 保存预测之后的概率值
    result_file_name = result_path + str(model_name) + '_features_' + str(len_feature_choose) + '_proba.csv'
    result_data = pd.concat(
        [data_predict_user_id, pd.DataFrame(predict_proba)], axis=1)
    to_csv(result_data, result_file_name)

    # 保存预测结果，卡了阈值之后的结果

    result_file_name = result_path + str(model_name) + '_features_' + str(
        len_feature_choose) + '_proba_to_label_using_th_' + str(threshold) + '.csv'
    result_data = pd.concat(
        [data_predict_user_id, pd.DataFrame(predict_label)], axis=1)
    to_csv(result_data, result_file_name)


def feature_preprocessor(data_train, data_predict, label_name):
    """
    数据预处理
    :return:
    """
    data_train_without_label = data_train.drop(label_name, axis=1)

    feature_name = list(data_train_without_label.columns.values)
    data_predict_user_id = data_predict['uid']

    # 用test和train数据进行数值填充
    data_all = pd.concat([data_train_without_label, data_predict])

    data_train_filled = data_train_without_label.fillna(
        value=data_all.median())

    print(base_info(data_train))
    print(base_describe(data_train))
    data_train.dropna(axis=1)

    # 特征构建
    x_temp = data_train_filled.iloc[:, :].values
    y = data_train.iloc[:, -1].values

    # 特征选择
    X, feature_score_dict_sorted, feature_used_name, feature_not_used_name = feature_select(
        '', x_temp, y, feature_name)

    print(feature_used_name)
    print(feature_not_used_name)
    dropped_feature_name = feature_not_used_name
    dropped_feature_name.append('uid')
    len_feature_choose = len(feature_used_name)
    data_predict_filled = data_predict.fillna(value=data_all.median())
    X_test = columns_drop(data_predict_filled, dropped_feature_name).values

    return X, y, X_test, len_feature_choose, data_predict_user_id


if __name__ == "__main__":
    base_data_path = '/Users/weijinqian/Documents/feature/feature_eda/data/project'
    result_path = base_data_path + '/model_result/'
    model_path = base_data_path + '/model/'
    train_data_path = base_data_path + '/train_data.txt'
    # 随机整一批用来预测的数据
    predict_data_path = base_data_path + '/predict_data.txt'
    train_data = pd.read_csv(train_data_path, sep='\t')
    train_data = shuffle(train_data)
    train_data = train_data.drop('uid', axis=1)
    train_data['label'] = train_data['label'].apply(lambda x: 1 if x > 0.25 else 0)
    print(train_data.columns)

    # 预测数据
    predict_data = pd.read_csv(predict_data_path, sep='\t')
    predict_data = predict_data[get_model_features()]
    print(predict_data.columns)

    train_model(train_data, predict_data, result_path, model_path=model_path)
