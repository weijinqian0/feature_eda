import datetime
import pickle

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import log_loss, classification_report
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
from sklearn.utils.multiclass import type_of_target
from sklearn.metrics import f1_score

from src.model.model_interface import IModel
from abc import ABCMeta, abstractmethod


class BaseModel(metaclass=ABCMeta):
    """
        逻辑回归模型
        """

    def __init__(self) -> None:
        self.clf = self.get_origin_model()
        self.model_name = self.get_model_name()

    @abstractmethod
    def get_origin_model(self):
        pass

    @abstractmethod
    def get_model_name(self):
        pass

    def data_process(self, data):
        return data

    def fit(self, train_data, label):
        train_data = self.data_process(train_data)
        self._model = self.clf.fit(train_data, label)

    def predict(self, data, threshold=0.5):
        predict_proba = self.predict_proba(data)[:, 1]
        predict_label = []
        for i in range(len(predict_proba)):
            if predict_proba[i] > threshold:
                predict_label.append(1)
            else:
                predict_label.append(0)
        return predict_label, predict_proba

    def predict_and_metric(self, data, label):
        y = label
        y_pred = self.predict_proba(data)[:, 1]
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

    def predict_proba(self, data):
        data = self.data_process(data)
        return self._model.predict_proba(data)

    def metrics(self, data):
        super().metrics(data)

    def cv_val(self, data):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        for train_idx, val_idx in cv.split(X, y):
            self.fit(X[train_idx], y[train_idx])
            self.predict_and_metric(X[val_idx], y[val_idx])

    def save_model(self, model_path):
        """
        模型保存位置
        :param model_path:
        :return:
        """
        ver = datetime.datetime.now().strftime('%Y-%m-%d%H:%M:%S')
        model_path += (self.model_name + '_' + str(ver) + '.pkl')
        BaseModel.save_model_with_pickle(self._model, model_path)

    def load_model(self, model_path):
        return BaseModel.load_model_with_pickle(model_path)

    @staticmethod
    def save_model_with_pickle(model, model_path):
        """
            保存下模型
            """
        with open(model_path, 'wb') as wf:
            pickle.dump(model, wf)

    @staticmethod
    def load_model_with_pickle(model_path):
        """
        返回加载的模型
        """
        with open(model_path, 'rb') as f:
            return pickle.load(f)


class LrClassifier(BaseModel):
    """
    逻辑回归模型
    """

    def get_origin_model(self):
        return LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')

    def get_model_name(self):
        return 'lr'

    def data_process(self, data):
        """
        数据归一化
        :param data:
        :return:
        """
        transfer = StandardScaler()
        return transfer.fit_transform(data)


class KnnClassifier(BaseModel):

    def get_origin_model(self):
        return KNeighborsClassifier(n_neighbors=6)

    def get_model_name(self):
        return 'knn_Classifier'

    def data_process(self, data):
        """
        数据归一化
        :param data:
        :return:
        """
        transfer = StandardScaler()
        return transfer.fit_transform(data)


class GaussianNBClassifier(BaseModel):

    def get_origin_model(self):
        return GaussianNB()

    def get_model_name(self):
        return 'GaussianNB'

    def data_process(self, data):
        """
        数据归一化
        :param data:
        :return:
        """
        transfer = StandardScaler()
        return transfer.fit_transform(data)


class DTClassifier(BaseModel):

    def get_origin_model(self):
        return DecisionTreeClassifier(n_neighbors=3)

    def get_model_name(self):
        return 'DecisionTreeClassifier'

    def data_process(self, data):
        """
        数据归一化
        :param data:
        :return:
        """
        transfer = StandardScaler()
        return transfer.fit_transform(data)


class RFClassifier(BaseModel):

    def get_origin_model(self):
        return RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)

    def get_model_name(self):
        return 'RandomForestClassifier'

    def data_process(self, data):
        """
        数据归一化
        :param data:
        :return:
        """
        transfer = StandardScaler()
        return transfer.fit_transform(data)


class ETClassifier(BaseModel):

    def get_origin_model(self):
        return ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)

    def get_model_name(self):
        return 'RandomForestClassifier'

    def data_process(self, data):
        """
        数据归一化
        :param data:
        :return:
        """
        transfer = StandardScaler()
        return transfer.fit_transform(data)


class AdaClassifier(BaseModel):

    def get_origin_model(self):
        return AdaBoostClassifier(n_estimators=100)

    def get_model_name(self):
        return 'AdaBoostClassifier'

    def data_process(self, data):
        """
        数据归一化
        :param data:
        :return:
        """
        transfer = StandardScaler()
        return transfer.fit_transform(data)


class LgbmClassifier(BaseModel):

    def get_origin_model(self):
        return lgb

    def get_model_name(self):
        return 'lgbm'

    def data_process(self, data):
        pass

    def fit(self, train_data, label):
        print(type(train_data))
        print(type(label))
        train_data, test_data, train_target, test_target = train_test_split(train_data,
                                                                            label,
                                                                            test_size=0.2,
                                                                            random_state=0)
        print(train_target)
        print(type(train_target))
        train_matrix = self.clf.Dataset(train_data, label=pd.Series(train_target))
        test_matrix = self.clf.Dataset(test_data, label=pd.Series(test_target))
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

        num_round = 100
        early_stopping_rounds = 100
        self._model = self.clf.train(params, train_matrix, num_round, valid_sets=test_matrix,
                                     early_stopping_rounds=early_stopping_rounds)

    def predict_proba(self, data):
        return self._model.predict(data, num_iteration=self._model.best_iteration)

    def save_model(self, model_path):
        ver = datetime.datetime.now().strftime('%Y-%m-%d%H:%M:%S')
        model_path += (self.model_name + '_' + str(ver) + '.m')
        self._model.save_model(model_path)

    def load_model(self, model_path):
        return self.clf.Booster(model_file=model_path)


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
