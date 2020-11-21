import warnings

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

plt.rcParams.update({'figure.max_open_warning': 0})
import seaborn as sns
from sklearn.model_selection import train_test_split, RepeatedKFold, GridSearchCV, cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error
import numpy as np
import pandas as pd

"""
1. 导入数据；
2. 合并数据；
3. 删除相关特征
4. 数据归一化
5. 画图探索特征和标签的关系
6. Box_cox变换
7. 分位数计算和绘图
8. 标签数据对数变换，使得数据更符合正太分布
"""


def get_training_data(data_all):
    # extract training samples
    df_train = data_all[data_all["origin"] == "train"]
    df_train['label'] = data_all.target1
    # split salePrice and features
    y = df_train.target
    X = df_train.drop(['origin', 'target', 'label'], axis=1)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=100)

    return X_train, X_valid, y_train, y_valid


# extract test data(without SalePrice)
def get_test_data(data_all):
    df_test = data_all[data_all['origin'] == "test"].reset_index(drop=True)
    return df_test.drop(['origin', 'target'], axis=1)


def rmse(y_true, y_pred):
    diff = y_pred - y_true
    sum_sq = sum(diff ** 2)
    n = len(y_pred)

    return np.sqrt(sum_sq / n)


def mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)


rmse_scorer = make_scorer(rmse, greater_is_better=False)
mse_scorer = make_scorer(mse, greater_is_better=False)

from sklearn.preprocessing import StandardScaler


def get_training_data_omitoutliers():
    return [], []


def train_model(model, param_grid=[], X=[], y=[], splits=5, repeats=5):
    # get unmodified training data, unless data to use already specified
    if len(y) == 0:
        X, y = get_training_data_omitoutliers()
    # create cross_validation method
    rkfold = RepeatedKFold(n_splits=splits, n_repeats=repeats)

    # perform a grid search if param_grid given
    if len(param_grid) > 0:
        gsearch = GridSearchCV(model, param_grid, cv=rkfold,
                               scoring='neg_mean_squared_error',
                               verbose=1,
                               return_train_score=True)
        # search the grid
        gsearch.fit(X, y)
        # extract best model from the grid
        model = gsearch.best_estimator_
        best_idx = gsearch.best_index_

        # get cv_scores for best model
        grid_results = pd.DataFrame(gsearch.cv_results_)
        cv_mean = abs(grid_results.loc[best_idx, 'mean_test_score'])
        cv_std = grid_results.loc[best_idx, 'std_test_score']
    else:
        grid_results = []
        cv_results = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=rkfold)

        cv_mean = abs(np.mean(cv_results))
        cv_std = np.std(cv_results)
    # combine mean and std cv-score in to pandas series
    cv_score = pd.Series({'mean': cv_mean, 'std': cv_std})

    # predict y using the fitted model
    y_pred = model.predict(X)

    # print stats on model performance
    print('-------------------')
    print(model)
    print('-------------------')
    print('score=', model.score(X, y))
    print('rmse=', rmse(y, y_pred))
    print('mse=', mse(y, y_pred))
    print('cross_val:mean=', cv_mean, ', std=', cv_std)

    # residual plots
    y_pred = pd.Series(y_pred, index=y.index)
    resid = y - y_pred
    mean_resid = resid.mean()
    std_resid = resid.std()
    z = (resid - mean_resid) / std_resid
    n_outliers = sum(abs(z) > 3)

    plt.figure(figsize=(15, 5))
    ax_131 = plt.subplot(1, 3, 1)
    plt.plot(y, y_pred, '.')
    plt.xlabel('y')
    plt.ylabel('y_pred')
    plt.title('corr=[:.3f]'.format(np.corrcoef(y, y_pred)[0][1]))
    ax_132 = plt.subplot(1, 3, 2)
    plt.plot(y, y - y_pred, '.')
    plt.xlabel('y')
    plt.ylabel('y - y_pred')
    plt.title('std resid = {:.3f}'.format(std_resid))

    ax_133 = plt.subplot(1, 3, 3)
    z.plot.hist(bins=50, ax=ax_133)
    plt.xlabel('z')
    plt.title('{:.0f} samples with z>3'.format(n_outliers))

    return model, cv_score, grid_results


opt_models = dict()
score_models = pd.DataFrame(columns=['mean', 'std'])
splits = 5
repeats = 5


def ridge():
    model = 'Ridge'
    opt_models[model] = Ridge()
    alph_range = np.arange(0.25, 6, 0.25)
    param_grid = {'alpha': alph_range}
    opt_models[model], cv_score, grid_results = train_model(opt_models[model], param_grid=param_grid, splits=splits,
                                                            repeats=repeats)
    cv_score.name = model

    plt.figure()
    plt.errorbar(alph_range, abs(grid_results['mean_test_score']),
                 abs(grid_results['std_test_score']) / np.sqrt(splits * repeats))
    plt.xlabel('alpha')
    plt.ylabel('score')

    return cv_score


def lasso():
    model = 'Ridge'
    opt_models[model] = Lasso()
    alph_range = np.arange(1e-4, 1e-3, 4e-5)
    param_grid = {'alpha': alph_range}
    opt_models[model], cv_score, grid_results = train_model(opt_models[model], param_grid=param_grid, splits=splits,
                                                            repeats=repeats)
    cv_score.name = model

    plt.figure()
    plt.errorbar(alph_range, abs(grid_results['mean_test_score']),
                 abs(grid_results['std_test_score']) / np.sqrt(splits * repeats))
    plt.xlabel('alpha')
    plt.ylabel('score')

    return cv_score


def elastic_net():
    model = 'ElasticNet'
    opt_models[model] = ElasticNet()

    param_grid = {'alpha': np.arange(1e-4, 1e-3, 1e-4), 'l1_ratio': np.arange(0.1, 1.0, 0.1), 'max_iter': [100000]}
    opt_models[model], cv_score, grid_results = train_model(opt_models[model], param_grid=param_grid, splits=splits,
                                                            repeats=repeats)
    cv_score.name = model

    return cv_score


def svr():
    model = 'LinearSVR'
    opt_models[model] = LinearSVR()
    crange = np.arange(0.1, 1.0, 0.1)
    param_grid = {'C': crange, 'max_iter': [1000]}
    opt_models[model], cv_score, grid_results = train_model(opt_models[model], param_grid=param_grid, splits=splits,
                                                            repeats=repeats)
    cv_score.name = model

    plt.figure()
    plt.errorbar(crange, abs(grid_results['mean_test_score']),
                 abs(grid_results['std_test_score']) / np.sqrt(splits * repeats))
    plt.xlabel('alpha')
    plt.ylabel('score')
    return cv_score


def knn():
    model = 'KNeighbors'
    opt_models[model] = KNeighborsRegressor()
    param_grid = {'n_neighboors': np.arange(3, 11, 1)}
    opt_models[model], cv_score, grid_results = train_model(opt_models[model], param_grid=param_grid, splits=splits,
                                                            repeats=repeats)
    cv_score.name = model

    plt.figure()
    plt.errorbar(np.arange(3, 11, 1), abs(grid_results['mean_test_score']),
                 abs(grid_results['std_test_score']) / np.sqrt(splits * repeats))
    plt.xlabel('alpha')
    plt.ylabel('score')
    return cv_score


# 使用boosting方法

def gbdt():
    model = 'GradientBoosting'
    opt_models[model] = GradientBoostingRegressor()
    param_grid = {'n_estimators': [150, 250, 350],
                  'max_depth': [1, 2, 3],
                  'min_samples_split': [5, 6, 7]}
    opt_models[model], cv_score, grid_results = train_model(opt_models[model], param_grid=param_grid, splits=splits,
                                                            repeats=repeats)
    cv_score.name = model

    return cv_score


def xgb():
    model = 'XGB'
    opt_models[model] = XGBRegressor(objective='reg:squarederror')
    param_grid = {'n_estimators': [100, 200, 300, 400, 500],
                  'max_depth': [1, 2, 3]
                  }
    opt_models[model], cv_score, grid_results = train_model(
        opt_models[model],
        param_grid=param_grid,
        splits=splits,
        repeats=repeats
    )
    cv_score.name = model

    return cv_score


# bagging 融合方法

def random_forest():
    model = 'RandomForest'
    opt_models[model] = RandomForestRegressor()
    param_grid = {
        'n_estimators': [100, 150, 200],
        'max_features': [8, 12, 16, 20, 24],
        'min_samples_split': [2, 4, 6]
    }
    opt_models[model], cv_score, grid_results = train_model(
        opt_models[model],
        param_grid=param_grid,
        splits=splits,
        repeats=repeats
    )
    cv_score.name = model

    return cv_score


def model_predict(test_data, test_y=[], stack=False):
    i = 0
    y_predict_total = np.zeros((test_data.shape[0],))
    for model in opt_models.keys():
        if model != "LinearSVR" and model != "KNeighbors":
            y_predict = opt_models[model].predict(test_data)
            y_predict_total += y_predict
            i += 1
        if len(test_y) > 0:
            print("{}_mse".format(model), mean_squared_error(y_predict, test_y))

        y_predict_mean = np.round(y_predict_total / i, 3)
        if len(test_y) > 0:
            print("mean_mse", mean_squared_error(y_predict_mean, test_y))
        else:
            y_predict_mean = pd.Series(y_predict_mean)
            return y_predict_mean





if __name__ == "__main__":
    score_models.append(ridge())
    score_models.append(lasso())
