from itertools import cycle

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from src.feature_handle.base_utils import *
from src.feature_handle.feature_embedded import xgb_embedded
from src.feature_handle.view_utils import paint_auc_curve


def xgb_lgb_cv_model(data_train, data_predict, result_path):
    """
    提供一键调用方式
    :param result_path: 结果保存位置
    :param data_train:
    :param data_predict:
    :return:
    """
    data_train_without_label = data_train.drop('Label', axis=1)

    feature_name = list(data_train_without_label.columns.values)
    data_predict_user_id = data_predict['uid']

    # 用test和train数据进行数值填充
    data_all = pd.concat([data_train_without_label, data_predict])
    data_train_filled = data_train_without_label.fillna(value=data_all.median())

    # 特征构建
    x_temp = data_train_filled.iloc[:, :].as_matrix()
    y = data_train.iloc[:, -1].as_matrix()

    # 特征选择
    X, feature_score_dict_sorted, feature_used_name, feature_not_used_name = xgb_embedded(feature_name, x_temp, y,
                                                                                          '0.1*mean')
    dropped_feature_name = feature_not_used_name
    len_feature_choose = len(feature_used_name)

    data_predict_filled = data_predict.fillna(value=data_all.median())
    data_predict_filled_after_feature_selection = columns_drop(data_predict_filled, dropped_feature_name)

    '''Split train/test data sets'''
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)  # 分层抽样  cv的意思是cross-validation

    '''Choose a classification model'''
    parameter_n_estimators = 100
    classifier = LGBMClassifier(n_estimators=parameter_n_estimators, learning_rate=0.1)

    '''hyperparameter optimization'''
    # param = {
    #     'max_depth': 6,
    #     'num_leaves': 64,
    #     'learning_rate': 0.03,
    #     'scale_pos_weight': 1,
    #     'num_threads': 40,
    #     'objective': 'binary',
    #     'bagging_fraction': 0.7,
    #     'bagging_freq': 1,
    #     'min_sum_hessian_in_leaf': 100
    # }
    #
    # param['is_unbalance'] = 'true'
    # param['metric'] = 'auc'

    # （1）num_leaves
    #
    # LightGBM使用的是leaf - wise的算法，因此在调节树的复杂程度时，使用的是num_leaves而不是max_depth。
    #
    # 大致换算关系：num_leaves = 2 ^ (max_depth)
    #
    # （2）样本分布非平衡数据集：可以param[‘is_unbalance’]=’true’
    #
    # （3）Bagging参数：bagging_fraction + bagging_freq（必须同时设置）、feature_fraction
    #
    # （4）min_data_in_leaf、min_sum_hessian_in_leaf

    # Model fit, predict and ROC
    colors = cycle(['cyan', 'indigo', 'seagreen', 'orange', 'blue'])
    lw = 2
    mean_f1 = 0.0
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 500)
    i_of_roc = 0
    threshold = 0.5
    probs = []
    label = []

    for (train_idx, test_idx), color in zip(cv.split(X, y), colors):
        a_model = classifier.fit(X[train_idx], y[train_idx])
        proba = a_model.predict_proba(X[test_idx])
        paint_auc_curve(y[test_idx], proba[:, 1], color, i_of_roc)
        label.extend(y[test_idx])
        probs.extend(proba[:, 1])
        i_of_roc += 1
        predict_label = proba[:, 1]
        for i in range(len(predict_label)):
            if predict_label[i] > threshold:
                predict_label[i] = 1
            else:
                predict_label[i] = 0
        lt = predict_label.astype('int32')
        f1 = f1_score(y[test_idx], lt)
        mean_f1 += f1
    paint_auc_curve(label, probs, 'blue', 'mean')

    a_model = classifier.fit(X, y)

    proba_predict = a_model.predict_proba(data_predict_filled_after_feature_selection)

    # 保存预测之后的概率值
    result_file_name = result_path + str(parameter_n_estimators) + '_features_' + str(
        len_feature_choose) + '_proba.csv'
    result_data = pd.concat([data_predict_user_id, pd.DataFrame(proba_predict[:, 1].tolist())], axis=1)
    to_csv(result_data, result_file_name)

    # 保存预测结果，卡了阈值之后的结果
    predict_label = proba_predict[:, 1]
    for i in range(len(predict_label)):
        if predict_label[i] > threshold:
            predict_label[i] = 1
        else:
            predict_label[i] = 0
    lt = predict_label.astype('int32')
    result_file_name = result_path + str(parameter_n_estimators) + '_features_' + str(
        len_feature_choose) + '_proba_to_label_using_th_' + str(threshold) + '.csv'
    result_data = pd.concat([data_predict_user_id, pd.DataFrame(lt.tolist())], axis=1)
    to_csv(result_data, result_file_name)
