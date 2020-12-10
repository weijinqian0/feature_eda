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


def get_model_features():
    return [
        'uid', 'comment_count', 'comment_words_max',
        'comment_words_mean', 'comment_words_min', 'comment_words_std',
        'type_ratio', 'video_count', 'pictext_count', 'other_count',
        'time_interval_max', 'time_interval_mean', 'time_interval_min',
        'time_interval_std', 'comment_score_max', 'comment_score_mean',
        'comment_score_min', 'comment_score_std', 'comment_score_2',
        'comment_score_1', 'comment_score_0', 'percentage_2', 'percentage_1']


def feature_select(name, train_data, target, feature_name):
    if name == "lgb":
        return lgb_embeded_1(train_data, target, feature_name, '0.1*mean')
    else:
        return train_data, target, feature_name, []


def model(name):
    if name == 'lr':
        return LogisticRegression(random_state=0, solver='lbfgs',
                                  multi_class='multinomial')
    elif name == 'lgb':
        return LGBMClassifier(
            n_estimators=100, learning_rate=0.1)
    pass


def lr_model(train_data, train_target, test_data, test_target):
    clf = LogisticRegression(random_state=0, solver='lbfgs',
                             multi_class='multinomial').fit(train_data, train_target)
    clf.score(test_data, test_target)


def cls_metrics(y, y_pred):
    for throd in [0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9]:
        print("lr evaluate throd:", throd)
        y_pred_binary = (y_pred >= throd) * 1
        print('AUC: %.4f' % metrics.roc_auc_score(y, y_pred_binary))
        print('ACC: %.4f' % metrics.accuracy_score(y, y_pred_binary))
        print('Recall: %.4f' % metrics.recall_score(
            y, y_pred_binary))  # average="micro"
        print('Precesion: %.4f' %
              metrics.precision_score(y, y_pred_binary))
        print('F1-score: %.4f' % metrics.f1_score(y, y_pred_binary))
        print(classification_report(y, y_pred_binary))


def xgb_lgb_cv_model(data_train, data_predict, result_path, model_path, threshold=0.5, label='label', is_save_model=True):
    """
    提供一键调用方式
    :param result_path: 结果保存位置
    :param data_train:
    :param data_predict:
    :return:
    """
    data_train_without_label = data_train.drop(label, axis=1)

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
    x_temp = data_train_filled.iloc[:, :].as_matrix()
    y = data_train.iloc[:, -1].as_matrix()

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

    '''Split train/test data sets'''
    cv = StratifiedKFold(n_splits=5, shuffle=True,
                         random_state=0)  # 分层抽样  cv的意思是cross-validation

    '''Choose a classification model'''
    model_name = 'lr'
    if model_name == 'lr':
        transfer = StandardScaler()
        X = transfer.fit_transform(X)
        X_test = transfer.fit_transform(X_test)
    classifier = model(model_name)

    # for train_idx, val_idx in cv.split(X, y):
    #     a_model = classifier.fit(X[train_idx], y[train_idx])
    #     proba = a_model.predict_proba(X[val_idx])
    #     cls_metrics(y[val_idx], proba[:, 1])

    a_model = classifier.fit(X, y)
    if is_save_model:
        ver = datetime.datetime.now().strftime('%Y-%m-%d%H:%M:%S')
        model_path += (model_name+'_'+str(ver)+'.pkl')
        save_model(a_model, model_path)

    proba_predict = a_model.predict_proba(X_test)
    # cls_metrics(y, proba_predict[:, 1])

    # 保存预测之后的概率值
    result_file_name = result_path + str(model_name) + '_features_' + str(
        len_feature_choose) + '_proba.csv'
    result_data = pd.concat(
        [data_predict_user_id, pd.DataFrame(proba_predict[:, 1].tolist())], axis=1)
    to_csv(result_data, result_file_name)

    # 保存预测结果，卡了阈值之后的结果
    predict_label = proba_predict[:, 1]
    for i in range(len(predict_label)):
        if predict_label[i] > threshold:
            predict_label[i] = 1
        else:
            predict_label[i] = 0
    lt = predict_label.astype('int32')
    result_file_name = result_path + str(model_name) + '_features_' + str(
        len_feature_choose) + '_proba_to_label_using_th_' + str(threshold) + '.csv'
    result_data = pd.concat(
        [data_predict_user_id, pd.DataFrame(lt.tolist())], axis=1)
    to_csv(result_data, result_file_name)


def save_model(clf, model_path):
    """
    保存下模型
    """
    with open(model_path, 'wb') as wf:
        pickle.dump(clf, wf)


def load_model(model_path):
    """
    返回加载的模型
    """
    with open('save/clf.pickle', 'rb') as f:
        clf = pickle.load(f)
        return clf


if __name__ == "__main__":
    base_data_path = '/home/disk4/weijinqian/baidu/personal-code/comsumer-interests/data/consumer_profile/baidu_app'
    result_path = base_data_path+'/model_result/'
    model_path = base_data_path+'/model/'
    train_data_path = base_data_path+'/train_data.txt'
    # 随机整一批用来预测的数据
    predict_data_path = base_data_path+'/predict_data.txt'
    train_data = pd.read_csv(train_data_path, sep='\t')
    train_data = shuffle(train_data)
    train_data = train_data.drop('uid', axis=1)

    # 预测数据
    predict_data = pd.read_csv(predict_data_path, sep='\t')
    predict_data = predict_data[get_model_features()]

    xgb_lgb_cv_model(train_data, predict_data,
                     result_path, model_path=model_path)
    pass
