from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import log_loss
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
import lightgbm as lgb
import xgboost
from sklearn.datasets import make_classification
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import make_gaussian_quantiles
from sklearn import metrics
from sklearn.metrics import f1_score


def lr_model(train_data, train_target, test_data, test_target):
    clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(train_data, train_target)
    clf.score(test_data, test_target)


def knn_model(train_data, train_target, test_data, test_target):
    clf = KNeighborsClassifier(n_neighbors=3).fit(train_data, train_target)
    clf.score(test_data, test_target)


def gnb_model(train_data, train_target, test_data, test_target):
    clf = GaussianNB().fit(train_data, train_target)
    clf.score(test_data, test_target)


def decision_tree_model(train_data, train_target, test_data, test_target):
    clf = DecisionTreeClassifier()
    clf.fit(train_data, train_target)
    clf.score(test_data, test_target)


def random_foreat_model(train_data, train_target, test_data, test_target):
    clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
    clf = clf.fit(train_data, train_target)

    clf.score(test_data, test_target)


def et_model(train_data, train_target, test_data, test_target):
    clf = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
    clf = clf.fit(train_data, train_target)
    clf.score(test_data, test_target)


def ada_model(train_data, train_target, test_data, test_target):
    clf = AdaBoostClassifier(n_estimators=100)
    clf = clf.fit(train_data, train_target)
    clf.score(test_data, test_target)


def gbdt_model(train_data, train_target, test_data, test_target):
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
    clf = clf.fit(train_data, train_target)
    clf.score(test_data, test_target)


def ensemble_model(train_data, train_target, test_data, test_target):
    clf1 = LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=1)
    clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
    clf3 = GaussianNB()

    eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')
    for clf, label in zip([clf1, clf2, clf3, eclf],
                          ['Logistic Regression', 'Random Forest', 'naive Bayes', 'Ensemble']):
        scores = cross_val_score(clf, train_data, train_target, cv=5, scoring='accuracy')
        print("Accuracy: %0.2f(+/-%0.2f)[%s]" % (scores.mean(), scores.std(), label))


def lgb_model(train_data, train_target, test_data, test_target):
    clf = lgb
    train_matrix = clf.Dataset(train_data, label=train_target)
    test_matrix = clf.Dataset(test_data, label=test_target)

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
    early_stopping_rounds = 1000
    model = clf.train(params, train_matrix, num_round, valid_sets=test_matrix,
                      early_stopping_rounds=early_stopping_rounds)

    pre = model.predict(test_matrix, num_iteration=model.best_iteration)


def xgb_model(train_data, train_target, test_data, test_target, valid_data, valid_target):
    clf = xgboost
    train_matrix = clf.DMatrix(train_data, label=train_target, missing=-1)
    test_matrix = clf.DMatrix(test_data, label=test_target, missing=-1)
    z = clf.DMatrix(valid_data, label=valid_target, missing=-1)

    params = {
        'booster': 'gbtree',
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'gamma': 1,
        'min_child_weight': 1.5,
        'max_depth': 5,
        'lambda': 10,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'colsample_bylevel': 0.7,
        'eta': 0.03,
        'tree_method': 'exact',
        'seed': 2017,
        'num_class': 2,
    }

    num_round = 10000
    early_stopping_rounds = 1000
    watchlist = [(train_matrix, 'train'), (test_matrix, 'eval')]
    model = clf.train(params, train_matrix, num_round, evals=watchlist,
                      early_stopping_rounds=early_stopping_rounds)

    pre = model.predict(test_matrix, num_iteration=model.best_iteration)


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold


class SBBTree():
    """
    SBBTree
    Stacking,Bootstap,Bagging
    """

    def __init__(self, params, stacking_num, bagging_num, bagging_test_size, num_boost_round,
                 early_stopping_rounds) -> None:
        self.params = params
        self.stacking_num = stacking_num
        self.bagging_num = bagging_num
        self.bagging_test_size = bagging_test_size
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds

        self.model = lgb
        self.stacking_model = []
        self.bagging_model = []

    def fit(self, X, y):
        if self.stacking_num > 1:
            layer_train = np.zeros((X.shape[0], 2))
            self.SK = StratifiedKFold(n_splits=self.stacking_num, shuffle=True, random_state=1)
            for k, (train_index, test_index) in enumerate(self.SK.split(X, y)):
                X_train = X[train_index]
                y_train = y[train_index]
                X_test = X[test_index]
                y_test = y[test_index]

                lgb_train = lgb.Dataset(X_train, y_train)
                lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

                gbm = lgb.train(
                    self.params,
                    lgb_train,
                    num_boost_round=self.num_boost_round,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=self.early_stopping_rounds
                )

                self.stacking_model.append(gbm)

                pred_y = gbm.predict(X_test, num_iteration=gbm.best_iteration)
                layer_train[test_index, 1] = pred_y

            X = np.hstack((X, layer_train[:, 1].reshape((-1, 1))))
        else:
            pass

        for bn in range(self.bagging_num):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.bagging_test_size, random_state=bn)
            lgb_train = lgb.Dataset(X_train, y_train)
            lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

            gbm = lgb.train(self.params, lgb_train, num_boost_round=10000, valid_sets=lgb_eval,
                            early_stopping_rounds=200)

            self.bagging_model.append(gbm)

    def predict(self, X_pred):
        """
        predict test data
        :param X_pred:
        :return:
        """
        if self.stacking_num > 1:
            test_pred = np.zeros((X_pred.shape[0], self.stacking_num))
            for sn, gbm in enumerate(self.stacking_model):
                pred = gbm.predict(X_pred, num_iteration=gbm.best_iteration)
                test_pred[:, sn] = pred
                X_pred = np.hstack((X_pred, test_pred.mean(axis=1).reshape((-1, 1))))
        else:
            pass

        for bn, gbm in enumerate(self.bagging_model):
            pred = gbm.predict(X_pred, num_iteration=gbm.best_iteration)

            if bn == 0:
                pred_out = pred
            else:
                pred_out += pred

        return pred_out / self.bagging_num


if __name__ == "__main__":
    # 测试自己封装的模型类
    # X, y = make_classification(n_samples=1000, n_features=25, n_clusters_per_class=1, n_informative=15, random_state=1)
    X, y = make_gaussian_quantiles(mean=None, cov=1.0, n_samples=1000, n_features=50, n_classes=2, shuffle=True,
                                   random_state=2)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': 9,
        'learning_rate': 0.03,
        'feature_fraction_seed': 2,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_data': 20,
        'min_hessian': 1,
        'verbose': -1,
        'silent': 0
    }
    # test 1
    model = SBBTree(params, stacking_num=2, bagging_num=1, bagging_test_size=0.33, num_boost_round=10000,
                    early_stopping_rounds=200)

    model.fit(X, y)
    X_pred = X[0].reshape((1, -1))
    pred = model.predict(X_pred)
    print('pred')
    print(pred)
    print('TEST 1 OK')

    # test 1
    model = SBBTree(params, stacking_num=1, bagging_num=1, bagging_test_size=0.33, num_boost_round=10000,
                    early_stopping_rounds=200)

    model.fit(X, y)
    X_pred = X[0].reshape((1, -1))
    pred1 = model.predict(X_pred)

    # test 2
    model = SBBTree(params, stacking_num=1, bagging_num=3, bagging_test_size=0.33, num_boost_round=10000,
                    early_stopping_rounds=200)

    model.fit(X, y)
    X_pred = X[0].reshape((1, -1))
    pred2 = model.predict(X_pred)

    # test 3
    model = SBBTree(params, stacking_num=5, bagging_num=1, bagging_test_size=0.33, num_boost_round=10000,
                    early_stopping_rounds=200)

    model.fit(X, y)
    X_pred = X[0].reshape((1, -1))
    pred3 = model.predict(X_pred)

    # test 1
    model = SBBTree(params, stacking_num=5, bagging_num=3, bagging_test_size=0.33, num_boost_round=10000,
                    early_stopping_rounds=200)

    model.fit(X, y)
    X_pred = X[0].reshape((1, -1))
    pred4 = model.predict(X_pred)

    fpr, tpr, thresholds = metrics.roc_curve(y_test + 1, pred1, pos_label=2)
    print('auc:', metrics.auc(fpr, tpr))

    fpr, tpr, thresholds = metrics.roc_curve(y_test + 1, pred2, pos_label=2)
    print('auc:', metrics.auc(fpr, tpr))

    fpr, tpr, thresholds = metrics.roc_curve(y_test + 1, pred3, pos_label=2)
    print('auc:', metrics.auc(fpr, tpr))

    fpr, tpr, thresholds = metrics.roc_curve(y_test + 1, pred4, pos_label=2)
    print('auc:', metrics.auc(fpr, tpr))

    # pred=model.predict(test)
    # df_out=pd.DataFrame()
    # df_out['user_id']=test_data['user_id']
    # df_out['predict_prob']=pred
    # df_out.head()
