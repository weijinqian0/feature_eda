from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, LeaveOneOut, LeavePOut
from sklearn.model_selection import train_test_split

"""
cross_validation
交叉验证
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


