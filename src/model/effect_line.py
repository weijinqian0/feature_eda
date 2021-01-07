"""
效果曲线，也就是学习曲线、验证曲线
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import learning_curve, validation_curve

plt.figure(figsize=(18, 10), dpi=50)


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    绘制学习曲线
    :param estimator: 模型
    :param title:
    :param X:
    :param y:
    :param ylim:
    :param cv:
    :param n_jobs:
    :param train_sizes:
    :return:
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)

    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                                                            train_sizes=train_sizes, scoring='neg_mean_squared_error')
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1,
                     color='r')

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1,
                     color='g')
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label="Training score")
    plt.plot(train_sizes, train_scores_mean, 'o-', color='g', label="Cross-validation score")

    plt.legend(loc='best')
    return plt


def plot_val_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    param_range = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.0000001]
    train_scores, test_scores = validation_curve(estimator, X, y, param_name='label', param_range=param_range, cv=cv,
                                                 scoring='r2', n_jobs=1)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title(title)
    plt.xlabel("alpha")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)

    plt.semilogx(param_range, train_scores_mean, label="training score", color='r')
    plt.fill_between(param_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.2,
                     color='r')

    plt.semilogx(param_range, test_scores_mean, label="Cross_validation score", color='g')
    plt.fill_between(param_range, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.2,
                     color='g')

    plt.legend(loc="best")
    plt.show()


if __name__ == "__main__":
    # X = train_data2[test_data2.columns].values
    # y = train_data2['target'].values
    title = "linearRegression"
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    # 这里estimator 只是个模型
    estimator = SGDRegressor()
    plot_learning_curve(estimator, title, X, y, ylim=(0.7, 1.01), cv=cv, n_jobs=1)
