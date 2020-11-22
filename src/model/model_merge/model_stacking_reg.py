"""
模型融合代码
"""
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
from scipy import sparse
import xgboost
import lightgbm

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# 基础代码
def stacking_reg(clf, train_x, train_y, test_x, folds, clf_name, kf, label_split=None):
    train = np.zeros((train_x.shape[0], 1))
    test = np.zeros((test_x.shape[0], 1))
    test_pre = np.empty((folds, test_x.shape[0], 1))

    cv_scores = []
    for i, (train_index, test_index) in enumerate(kf.split(train_x, label_split)):
        tr_x = train_x[train_index]
        tr_y = train_y[train_index]
        te_x = train_x[test_index]
        te_y = train_y[test_index]

        if clf_name in ["rf", 'ada', 'gb', 'et', 'lr', 'lsvc', 'knn']:
            clf.fit(tr_x, tr_y)
            pre = clf.predict(te_x).reshape(-1, 1)
            train[test_index] = pre
            test_pre[i, :] = clf.predict(test_x).reshape(-1, 1)
            cv_scores.append(mean_squared_error(te_y, pre))
        elif clf_name in ['xgb']:
            train_matrix = clf.DMatrix(tr_x, labal=tr_y, missing=-1)
            test_matrix = clf.DMatrix(te_x, label=te_y, missing=-1)
            z = clf.DMatrix(test_x, label=te_y, missing=-1)

            params = {
                'booster': 'gbtree',
                'eval_metric': 'rmse',
                'gama': 1,
                'min_child_weight': 1.5,
                'max_depth': 5,
                'lambda': 10,
                'subsample': 0.7,
                'colsample_bytree': 0.7,
                'colsample_bylevel': 0.7,
                'eta': 0.03,
                'tree_method': 'exact',
                'seed': 2017,
                'nthread': 12
            }

            num_round = 10000
            early_stopping_rounds = 10
            watchlist = [(train_matrix, 'train'), (test_matrix, 'eval')]
            if test_matrix:
                model = clf.train(
                    params,
                    train_matrix,
                    num_boost_round=num_round,
                    evals=watchlist,
                    early_stopping_rounds=early_stopping_rounds
                )

                pre = model.predict(
                    test_matrix,
                    ntree_limit=model.best_ntree_limit
                ).reshape(-1, 1)

                train[test_index] = pre
                test_pre[i, :] = model.predict(z, ntree_limit=model.best_ntree_limit).reshape(-1, 1)
        elif clf_name in ['lgb']:
            train_matrix = clf.Dataset(tr_x, label=tr_y)
            test_matrix = clf.Dataset(te_x, label=te_y)
            params = {
                'boosting_type': 'gbdt',
                'objective': 'regression_l2',
                'metric': 'mse',
                'min_child_weight': 1.5,
                'num_leaves': 2 ** 5,
                'lambda_l2': 10,
                'subsample': 0.7,
                'colsample_bytree': 0.7,
                'colsample_bylevel': 0.7,
                'learning_rate': 0.03,
                'tree_method': 'exact',
                'seed': 2017,
                'nthread': 12,
                'silent': True
            }
            num_round = 10000
            early_stopping_rounds = 100
            if test_matrix:
                model = clf.train(
                    params,
                    train_matrix,
                    num_round,
                    valid_sets=test_matrix,
                    early_stopping_rounds=early_stopping_rounds
                )

                pre = model.predict(te_x, num_iteration=model.best_iteration).reshape(-1, 1)
                train[test_index] = pre
                test_pre[i, :] = model.predict(test_x, num_iteration=model.best_iteration).reshape(-1, 1)
                cv_scores.append(mean_squared_error(te_y, pre))

        else:
            raise IOError("Please add new clf.")

        print("%s now score is:" % clf_name, cv_scores)
    test[:] = test_pre.mean(axis=0)
    print("%s_score_list" % clf_name, cv_scores)
    print("%s_score_mean:" % clf_name, np.mean(cv_scores))
    # 生成了一列
    return train.reshape(-1, 1), test.reshape(-1, 1)


def rf_reg(x_train, y_train, x_valid, kf, label_split=None):
    randomforest = RandomForestRegressor(
        n_estimators=600,
        max_depth=20,
        n_jobs=-1,
        random_state=2017,
        max_features='auto',
        verbose=1
    )

    rf_train, rf_test = stacking_reg(randomforest, x_train, y_train, x_valid, 'rf', kf, label_split=label_split)

    return rf_train, rf_test, "rf_reg"


def ada_reg(x_train, y_train, x_valid, kf, label_split=None):
    adaboost = AdaBoostRegressor(n_estimators=30, random_state=2017, learning_rate=0.01)

    ada_train, ada_test = stacking_reg(adaboost, x_train, y_train, x_valid, 'ada', kf, label_split=label_split)

    return ada_train, ada_test, 'ada_reg'


def gb_reg(x_train, y_train, x_valid, kf, label_split=None):
    gbdt = GradientBoostingRegressor(learning_rate=0.04,
                                     n_estimators=100,
                                     subsample=0.8,
                                     random_state=2017,
                                     max_depth=5,
                                     verbose=1
                                     )

    gbdt_train, gbdt_test = stacking_reg(gbdt, x_train, y_train, x_valid, 'gb', kf, label_split=label_split)
    return gbdt_train, gbdt_test, 'gb_reg'


def et_reg(x_train, y_train, x_valid, kf, label_split=None):
    extratree = ExtraTreesRegressor(n_estimators=600,
                                    max_depth=35,
                                    max_features='auto',
                                    n_jobs=-1,
                                    random_state=2017,
                                    verbose=1)

    et_train, et_test = stacking_reg(extratree, x_train, y_train, x_valid, 'et', kf, label_split=label_split)
    return et_train, et_test, 'et_reg'


def lr_reg(x_train, y_train, x_valid, kf, label_split=None):
    lr_reg = LinearRegression(n_jobs=1)
    lr_train, lr_test = stacking_reg(lr_reg, x_train, y_train, x_valid, 'lr', kf, label_split=label_split)
    return lr_train, lr_test, 'lr_reg'


def xgb_reg(x_train, y_train, x_valid, kf, label_split=None):
    xgb_train, xgb_test = stacking_reg(xgboost, x_train, y_train, x_valid, 'xgb', kf, label_split=label_split)
    return xgb_train, xgb_test, 'xgb_reg'


def lgb_reg(x_train, y_train, x_valid, kf, label_split=None):
    lgb_train, lgb_test = stacking_reg(lightgbm, x_train, y_train, x_valid, 'lgb', kf, label_split=label_split)
    return lgb_train, lgb_test, 'lgb_reg'


def stacking_pred(x_train, y_train, x_valid, kf, clf_list, label_split=None, clf_fin='lgb', if_concat_origin=True):
    for k, clf_list in enumerate(clf_list):
        clf_list = [clf_list]
        column_list = []
        train_data_list = []
        test_data_list = []

        for clf in clf_list:
            train_data, test_data, clf_name = clf(x_train, y_train, x_valid, kf, label_split=label_split)
            train_data_list.append(train_data)

            test_data_list.append(test_data)
            column_list.append("clf_%s" % (clf_name))

        train = np.concatenate(train_data_list, axis=1)
        test = np.concatenate(test_data_list, axis=1)

        if if_concat_origin:
            train = np.concatenate([x_train, train], axis=1)
            test = np.concatenate([x_valid, test], axis=1)

        print(x_train.shape)
        print(train.shape)
        print(clf_name)
        print(clf_name in ['lgb'])

        if clf_fin in ['rf', 'ada', 'gb', 'et', 'lr', 'lsvc', 'knn']:
            if clf_fin in ['rf']:
                clf = RandomForestRegressor(
                    n_estimators=600,
                    max_depth=20,
                    n_jobs=-1,
                    random_state=2017,
                    max_features='auto',
                    verbose=1
                )
            elif clf_fin in ['ada']:
                clf = AdaBoostRegressor(n_estimators=30, random_state=2017, learning_rate=0.01)
            elif clf_fin in ['gb']:
                clf = GradientBoostingRegressor(
                    learning_rate=0.04,
                    n_estimators=100,
                    subsample=0.8,
                    random_state=2017,
                    max_depth=5,
                    verbose=1
                )
            elif clf_fin in ['et']:
                clf = ExtraTreesRegressor(
                    n_estimators=600,
                    max_depth=35,
                    max_features='auto',
                    n_jobs=-1,
                    random_state=2017,
                    verbose=-1
                )
            elif clf_fin in ['lr']:
                clf = LinearRegression(n_jobs=-1)

            clf.fit(train, y_train)
            pre = clf.predict(test).reshape(-1, 1)
            return pre
        elif clf_fin in ['xgb']:
            clf = xgboost
            train_matrix = clf.DMatrix(train, label=y_train, missing=-1)
            test_matrix = clf.DMatrix(train, label=y_train, missing=-1)

            params = {
                'booster': 'gbtree',
                'eval_metric': 'rmse',
                'gamma': 1,
                'min_child_weight': 1.5,
                'max_depth': 5,
                'lambda': 10,
                'subsample': 0.7,
                'colsample_bytree': 0.7,
                'colsample_bylevel': 0.7,
                'eta': 0.03,
                'tree_method': 'extra',
                'seed': 2017,
                'nthread': 12
            }

            num_round = 10000
            early_stopping_rounds = 100
            watchlist = [(train_matrix, 'train'), (test_matrix, 'eval')]

            model = clf.train(params, train_matrix, num_boost_round=num_round, evals=watchlist,
                              early_stopping_rounds=early_stopping_rounds)

            pre = model.predict(test, ntree_limit=model.best_ntree_limit).reshape(-1, 1)

            return pre

        elif clf_fin in ['lgb']:
            print(clf_name)
            clf = lightgbm
            train_matrix = clf.Dataset(train, label=y_train)
            test_matrix = clf.Dataset(train, label=y_train)

            params = {
                'boosting_type': 'gbdt',
                'objective': 'regression_l2',
                'metric': 'mse',
                'min_child_weight': 1.5,
                'num_leaves': 2 ** 5,
                'lambda_l2': 10,
                'subsample': 0.7,
                'colsample_bytree': 0.7,
                'colsample_bylevel': 0.7,
                'learning_rate': 0.03,
                'tree_method': 'extract',
                'seed': 2017,
                'nthread': 12,
                'silent': True
            }

            num_round = 10000
            early_stopping_rounds = 100
            model = clf.train(
                params,
                train_matrix,
                num_round,
                valid_sets=test_matrix,
                early_stopping_rounds=early_stopping_rounds
            )

            pre = model.predict(test, num_iteration=model.best_iteration).reshape(-1, 1)

            return pre


if __name__ == "__main__":
    with open('data/zhengqi_train.txt') as rf:
        data_train = pd.read_table(rf, sep='\t')

    with open('data/zhengqi_test.txt') as rf_test:
        data_test = pd.read_table(rf_test, sep='\t')

    kf = KFold(n_splits=5, shuffle=True, random_state=0)

    x_train = data_train[data_test.columns].values
    x_valid = data_test[data_test.columns].values
    y_train = data_train['target'].values

    clf_list = [lr_reg, lgb_reg]

    pred = stacking_reg(x_train, y_train, x_valid, kf, clf_list, label_split=None, clf_fin='lgb', if_concat_origin=True)
