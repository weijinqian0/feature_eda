import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame

from sklearn.metrics import mean_squared_error, roc_curve, auc

matplotlib.use('TkAgg')
import seaborn as sns
from scipy import stats

import warnings

warnings.filterwarnings("ignore")
"""
各种特征视觉分析api调用
1. 单变量分析（统计量与变量的分布）
（1）箱型图；（2）distplot
2. 双变量分析
【1】连续和连续
    （1）绘制散点图（2）计算相关性
【2】类别和类别
    （1）双向表（2）堆叠柱状图
【3】类别和连续
    （1）小提琴图

"""

"""
单变量
"""


# 绘制箱型图，看单个
def paint_box(data, columns):
    """
    直接看指定的列的箱型图
    :param data:
    :param columns:
    :return:
    """
    length = len(columns)
    fig = plt.figure(figsize=(80, 60), dpi=75)
    for i in range(length):
        # 需要根据绘制的图片个数调整
        plt.subplot(4, 6, i + 1)
        sns.boxplot(data=pd.DataFrame(data[columns[i]]), orient="v", width=0.5)
        plt.ylabel(columns[i], fontsize=12)
    plt.show()


# 绘制直方图和QQ图 都是用来看数据分布是否是正太分布的
def paint_dist(data, columns, cols=6):
    rows = len(columns)
    plt.figure(figsize=(60, 20), dpi=50)

    i = 0
    for col in columns:
        # print(data[col])
        # if np.all(data[col]) == 0:
        #     continue
        i += 1
        ax = plt.subplot(7, 6, i)
        sns.distplot(data[col], fit=stats.norm)
        i += 1
        ax = plt.subplot(7, 6, i)
        res = stats.probplot(data[col], plot=plt)

    plt.tight_layout()
    plt.show()


def paint_dist_single(data, column, bins):
    """
    绘制单个变量的分布
    :param data:
    :param column:
    :param bins:可以是数字、序列。bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2]
    :return:
    """
    plt.hist(data[column], bins=bins)
    plt.show()


"""
双变量
"""


# 堆叠图
def paint_count(data, column_name, target):
    plt.figure(figsize=(8, 6))
    plt.title(str(column_name) + ' VS Label' + str(target))
    ax = sns.countplot(column_name, hue='label', data=data)


def paint_reg(train_data, column: str, y_name='target'):
    """
    绘制线性回归关系
    :param y_name:
    :param train_data:
    :param column: 训练数据所在的列名
    :return:
    """
    fools = 2
    frows = 1
    plt.figure(figsize=(8, 4), dpi=50)
    ax = plt.subplot(1, 2, 1)
    sns.regplot(x=column, y=y_name, data=train_data, ax=ax, scatter_kws={'marker': '.', 's': 3, 'alpha': 0.3},
                line_kws={'color': 'k'})
    plt.xlabel(column)
    plt.ylabel(y_name)

    ax = plt.subplot(1, 2, 2)
    sns.displot(train_data[column].dropna())
    plt.xlabel(column)

    plt.show()


def paint_heatmap(data: DataFrame, drop_columns, k):
    """
    绘制热力图
    :type k: 阈值k
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
    cols = data_corr.nlargest(k, 'target')['target'].index
    cm = np.corrcoef(data_1[cols].values.T)
    hm = plt.subplots(figsize=(10, 10))
    hm = sns.heatmap(data_1[cols].corr(), annot=True, square=True)
    plt.show()


# 小提琴图，用于分析类别和连续型变量
def paint_violin(data: DataFrame, column):
    plt.figure(figsize=[16, 10])
    sns.violinplot(x=data['label'], y=data[column])
    plt.show()


"""
异常值分析
"""


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


"""
训练数据和测试数据关系
"""


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


def paint_auc_curve(y_label, y_proba, color, index):
    """
    ROC 曲线绘制
    :param y_label:
    :param y_proba:
    :param color:
    :param index: 第几个图片
    :return:
    """
    fpr, tpr, thresholds = roc_curve(y_label, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, color=color, label='ROC fold %d (area = %0.4f)' % (index, roc_auc))
