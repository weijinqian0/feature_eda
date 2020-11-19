import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame

from sklearn.metrics import mean_squared_error

matplotlib.use('TkAgg')
import seaborn as sns
from scipy import stats

import warnings

warnings.filterwarnings("ignore")
"""
各种特征视觉分析api调用

"""


# 绘制箱型图
def paint_box(data, columns):
    length = len(columns)
    column = data.columns.tolist()[:length]
    fig = plt.figure(figsize=(80, 60), dpi=75)
    for i in range(length):
        plt.subplot(7, 8, i + 1)
        sns.boxplot(data[column[i]], orient="v", width=0.5)
        plt.ylabel(column[i], fontsize=36)
    plt.show()


#   基于模型预测来发现异常数据，看结果感觉还不错
def find_outliers(model, X, y, sigma=3):
    # 使用model预测y
    try:
        y_pred = pd.Series(model.predict(X), index=y.index)
    except:
        model.fit(X, y)
        y_pred = pd.Series(model.predict(X), index=y.index)

    resid = y - y_pred
    mean_resid = resid.mean()
    std_resid = resid.std()

    z = (resid - mean_resid) / std_resid
    outliers = z[abs(z) > sigma].index

    print('R2=', model.score(X, y))
    print('mse=', mean_squared_error(y, y_pred))
    print('-------------------------------------')

    print('mean of residuals:', mean_resid)
    print('std of residuals', std_resid)
    print('-------------------------------------')

    print(len(outliers), 'outliers:')
    print(outliers.tolist())

    plt.figure(figsize=(15, 5))
    ax_131 = plt.subplot(1, 3, 1)
    plt.plot(y, y_pred, '.')
    plt.plot(y.loc[outliers], y_pred.loc[outliers], 'ro')
    plt.legend(['Accepted', 'Outlier'])
    plt.xlabel('y')
    plt.ylabel('y_pred')

    ax_132 = plt.subplot(1, 3, 2)
    plt.plot(y, y - y_pred, '.')
    plt.plot(y.loc[outliers], y.loc[outliers] - y_pred.loc[outliers], 'ro')
    plt.legend('Accepted', 'Outlier')
    plt.xlabel('y')
    plt.ylabel('y - y_pred')

    ax_133 = plt.subplot(1, 3, 3)
    z.plot.hist(bins=50, ax=ax_133)
    z.loc[outliers].plot.hist(color='r', bins=50, ax=ax_133)
    plt.legend('Accepted', 'Outlier')
    plt.xlabel('z')

    plt.savefig('outliers.png')
    return outliers


# 绘制直方图和QQ图 都是用来看数据分布是否是正太分布的
def paint_dist(data, columns, cols=6):
    rows = len(columns)
    plt.figure(figsize=(4 * cols, 4 * rows))

    i = 0
    for col in columns:
        i += 1
        ax = plt.subplot(rows, cols, i)
        sns.distplot(data[col], fit=stats.norm)

        i += 1
        ax = plt.subplot(rows, cols, i)
        res = stats.probplot(data[col], plot=plt)

    plt.tight_layout()
    plt.show()


def paint_kde(train_data, test_data, column: str, y):
    """
    查看训练数据和测试数据，再某个特征下，分布是否一致
    :param train_data: 训练数据
    :param test_data: 测试数据
    :param column: 某一列特征
    :param y: 目标特征
    :return:
    """
    plt.figure(figsize=(8, 4), dpi=150)
    ax = sns.kdeplot(train_data[column], color="Red", shade=True)
    ax = sns.kdeplot(test_data[column], color='Blue', shade=True)
    ax.set_xlabel(column)
    ax.set_ylabel(y)
    ax = ax.legend('train', 'test')


def paint_reg(train_data, column: str, y):
    """
    绘制线性回归关系
    :param train_data:
    :param column:
    :param y:
    :return:
    """
    fools = 2
    frows = 1
    plt.figure(figsize=(8, 4), dpi=50)
    ax = plt.subplot(1, 2, 1)
    sns.regplot(x=column, y=y, data=train_data, ax=ax, scatter_kws={'marker': '.', 's': 3, 'alpha': 0.3},
                line_kws={'color': 'k'})
    plt.xlabel(column)
    plt.ylabel(y)

    ax = plt.subplot(1, 2, 2)
    sns.displot(train_data[column].dropna())
    plt.xlabel(column)

    plt.show()


def paint_heatmap(data: DataFrame, drop_columns):
    """
    绘制热力图
    :return:
    """
    pd.set_option("display.max_columns", 10)
    pd.set_option('display.max_rows', 10)
    if drop_columns is not None:
        data_1 = data.drop(drop_columns, axis=1)
    else:
        data_1 = data

    data_corr = data_1.corr()
    # 这个是整体的热力图
    # ax = plt.subplots(figsize=(20, 16))
    # ax = sns.heatmap(data_corr, vmax=.8, square=True, annot=True)
    # plt.show()

    # 这个是top10的热力图
    k = 10
    cols = data_corr.nlargest(k, 'target')['target'].index
    cm = np.corrcoef(data_1[cols].values.T)
    hm = plt.subplots(figsize=(10, 10))
    hm = sns.heatmap(data_1[cols].corr(), annot=True, square=True)
    plt.show()
