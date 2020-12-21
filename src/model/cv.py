from sklearn.metrics import mean_squared_error, confusion_matrix
from sklearn.model_selection import KFold, LeaveOneOut, LeavePOut, cross_val_score, cross_val_predict
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

"""
cross_validation
交叉验证
通过交叉验证看下模型在训练集上的效果
"""


def simple_cv(train, target, model):
    """
    简单的验证：直接将数据切分，然后验证效果
    :param train:
    :param target: label
    :param model:
    :return:
    """
    train_data, train_target, test_data, test_target = train_test_split(train, target, test_size=2.0, random_state=0)
    model.fit(train_data, train_target)
    score_train = mean_squared_error(train_target, model.predict(train_data))
    score_test = mean_squared_error(test_target, model.predict(test_data))
    print("score_train: ", score_train)
    print("score_test: ", score_test)


def kfold_cv(train, target, model):
    kf = KFold(n_splits=5)
    for k, (train_index, test_index) in enumerate(kf.split(train)):
        train_data, train_target, test_data, test_target = train.values[train_index], train[test_index], \
                                                           target[train_index], target[test_index]
        model.fit(train_data, train_target)
        score_train = mean_squared_error(train_target, model.predict(train_data))
        score_test = mean_squared_error(test_target, model.predict(test_data))
        print(k, "折", "score_train: ", score_train)
        print(k, "折", "score_test: ", score_test)


def leaveone_cv(train, target, model):
    """
    留一法交叉验证
    :param train:
    :param target:
    :param model:
    :return:
    """
    loo = LeaveOneOut()
    num = 100
    for k, (train_index, test_index) in enumerate(loo.split(train)):
        train_data, train_target, test_data, test_target = train.values[train_index], train[test_index], \
                                                           target[train_index], target[test_index]
        model.fit(train_data, train_target)
        score_train = mean_squared_error(train_target, model.predict(train_data))
        score_test = mean_squared_error(test_target, model.predict(test_data))
        print(k, "个", "score_train: ", score_train)
        print(k, "个", "score_test: ", score_test)


def leavep_cv(train, target, model):
    """
    留一法交叉验证
    :param train:
    :param target:
    :param model:
    :return:
    """
    loo = LeavePOut(p=10)
    num = 100
    for k, (train_index, test_index) in enumerate(loo.split(train)):
        train_data, train_target, test_data, test_target = train.values[train_index], train[test_index], \
                                                           target[train_index], target[test_index]
        model.fit(train_data, train_target)
        score_train = mean_squared_error(train_target, model.predict(train_data))
        score_test = mean_squared_error(test_target, model.predict(test_data))
        print(k, "10个", "score_train: ", score_train)
        print(k, "10个", "score_test: ", score_test)


def cv_score(train, target, model):
    """
    简单的cv验证打分
    :param train:
    :param target:
    :param model:
    :return:
    """
    scores = cross_val_score(model, train, target, scoring='neg_mean_squared_error', cv=10)
    scores = np.sqrt(-scores)
    display_scores(scores)


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("std", scores.std())


def display_confusion_matrix(train, target, model):
    """
    主要针对多分类的情形，直观观察混淆矩阵
    :return:
    """
    y_train_pred = cross_val_predict(model, train, target, cv=3)
    conf_mx = confusion_matrix(target, y_train_pred)
    plt.matshow(conf_mx)
    plt.show()
