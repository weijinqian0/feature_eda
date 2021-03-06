from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import lightgbm
import pandas as pd
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

"""
特征嵌入的方式
类型：
1. 基于惩罚项的特征选择；
2. 基于树模型的特征选择

"""


def penalty_embedded(train, target):
    """
    将带有L1惩罚项的逻辑回归作为基模型的特征选择
    L1惩罚项降维的原理在于保留多个对目标值具有同等相关性的特征中的一个，所以没选到的特征不代表不重要。故，可结合L2惩罚项来优化。
    具体操作为：若一个特征在L1中的权值为1，选择在L2中权值差别不大且在L1中权值为0的特征构成同类集合，将这一集合中的特征平分L1中的权值，
    故需要构建一个新的逻辑回归模型
    :param data:
    :param target:
    :return:
    """
    return SelectFromModel(LogisticRegression(penalty="l1", C=0.1)).fit_transform(train, target)


def tree_embedded(train, target, columns, threshold='mean'):
    """
    使用树模型gbdt进行特征选择
    补充特征选择的代码
    :param columns: 特征名称
    :param threshold: 其实就是selectFromModel的阈值
    :param train:
    :param target:
    :return:
    """
    # SelectFromModel
    clf = GradientBoostingClassifier(n_estimators=50, random_state=100)
    clf.fit(train, target)
    sfm = SelectFromModel(clf, prefit=True, threshold=threshold)
    # 这里其实就是已经执行了特征选择，直接返回选择之后的结果
    matrix_x = sfm.transform(train)

    # how much features whose feature importance is not zero
    feature_score_dict = {}
    for fn, s in zip(columns, clf.feature_importances_):
        feature_score_dict[fn] = s
    m = 0
    for k in feature_score_dict:
        if feature_score_dict[k] == 0.0:
            m += 1
    print('number of not-zero features:' + str(len(feature_score_dict) - m))

    # 特征重要度排序
    feature_score_dict_sorted = sorted(feature_score_dict.items(), key=lambda d: d[1], reverse=True)

    # 选择的特征
    how_long = matrix_x.shape[1]
    feature_used_dict_temp = feature_score_dict_sorted[:how_long]
    feature_used_name = []
    for ii in range(len(feature_used_dict_temp)):
        feature_used_name.append(feature_used_dict_temp[ii][0])

    # 没有选择的特征
    feature_not_used_name = []
    for i in range(len(columns)):
        if columns[i] not in feature_used_name:
            feature_not_used_name.append(columns[i])

    return matrix_x, feature_score_dict_sorted, feature_used_name, feature_not_used_name


def xgb_embedded(train, target, columns, threshold='mean'):
    """
        使用树模型gbdt进行特征选择
        补充特征选择的代码
        :param columns: 特征名称
        :param threshold: 其实就是selectFromModel的阈值
        :param train:
        :param target:
        :return:
        """
    # SelectFromModel
    clf = XGBClassifier(n_estimators=50, random_state=100)
    clf.fit(train, target)
    sfm = SelectFromModel(clf, prefit=True, threshold=threshold)
    # 这里其实就是已经执行了特征选择，直接返回选择之后的结果
    matrix_x = sfm.transform(train)

    # how much features whose feature importance is not zero
    feature_score_dict = {}
    for fn, s in zip(columns, clf.feature_importances_):
        feature_score_dict[fn] = s
    m = 0
    for k in feature_score_dict:
        if feature_score_dict[k] == 0.0:
            m += 1
    print('number of not-zero features:' + str(len(feature_score_dict) - m))

    # 特征重要度排序
    feature_score_dict_sorted = sorted(feature_score_dict.items(), key=lambda d: d[1], reverse=True)

    # 选择的特征
    how_long = matrix_x.shape[1]
    feature_used_dict_temp = feature_score_dict_sorted[:how_long]
    feature_used_name = []
    for ii in range(len(feature_used_dict_temp)):
        feature_used_name.append(feature_used_dict_temp[ii][0])

    # 没有选择的特征
    feature_not_used_name = []
    for i in range(len(columns)):
        if columns[i] not in feature_used_name:
            feature_not_used_name.append(columns[i])

    return matrix_x, feature_score_dict_sorted, feature_used_name, feature_not_used_name


def et_embedded(train, target, test):
    clf = ExtraTreesClassifier(n_estimators=50)
    clf.fit(train, target)
    model = SelectFromModel(clf, prefit=True)
    train_sel = model.transform(train)
    test_sel = model.transform(test)
    print('训练数据未特征筛选维度', train.shape)
    print('训练数据特征筛选维度', train_sel.shape)
    print(clf.feature_importances_[:10])


def lgb_embeded_1(train, target, columns, threshold='mean'):
    # SelectFromModel
    clf = LGBMClassifier(n_estimators=50, random_state=100)
    clf.fit(train, target)
    sfm = SelectFromModel(clf, prefit=True, threshold=threshold)
    # 这里其实就是已经执行了特征选择，直接返回选择之后的结果
    matrix_x = sfm.transform(train)

    # how much features whose feature importance is not zero
    feature_score_dict = {}
    for fn, s in zip(columns, clf.feature_importances_):
        feature_score_dict[fn] = s
    m = 0
    for k in feature_score_dict:
        if feature_score_dict[k] == 0.0:
            m += 1
    print('number of not-zero features:' + str(len(feature_score_dict) - m))

    # 特征重要度排序
    feature_score_dict_sorted = sorted(
        feature_score_dict.items(), key=lambda d: d[1], reverse=True)

    # 选择的特征
    how_long = matrix_x.shape[1]
    feature_used_dict_temp = feature_score_dict_sorted[:how_long]
    feature_used_name = []
    for ii in range(len(feature_used_dict_temp)):
        feature_used_name.append(feature_used_dict_temp[ii][0])

    # 没有选择的特征
    feature_not_used_name = []
    for i in range(len(columns)):
        if columns[i] not in feature_used_name:
            feature_not_used_name.append(columns[i])

    return matrix_x, feature_score_dict_sorted, feature_used_name, feature_not_used_name


def lgb_embeded(train, target, test, topK):
    X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=0.4, random_state=0)
    clf = lightgbm
    train_matrix = clf.Dataset(X_train, label=y_train)
    test_matrix = clf.Dataset(X_test, label=y_test)

    params = {
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'min_child_weight': 1.5,
        'num_leaves': 2 ** 5,
        'lambda_l2': 10,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'colsample_bylevel': 0.7,
        'learning_rate': 0.03,
        'tree_method': 'exact',
        'seed': 2017,
        'num_class': 2,
        'silent': True
    }
    num_round = 10000
    early_stopping_rounds = 100
    model = clf.train(params, train_matrix, num_round, valid_sets=test_matrix,
                      early_stopping_rounds=early_stopping_rounds)

    train_df = pd.DataFrame(train)
    train_df.columns = range(train.shape[1])

    test_df = pd.DataFrame(test)
    test_df.columns = range(test.shape[1])

    features_import = pd.DataFrame()
    features_import['importance'] = model.feature_importance()
    features_import['col'] = range(train.shape[1])

    features_import = features_import.sort_values(['importance'], ascending=0).head(topK)
    sel_col = list(features_import.columns)
    train_sel = train_df[sel_col]
    test_sel = test_df[sel_col]

    return train_sel, test_sel
