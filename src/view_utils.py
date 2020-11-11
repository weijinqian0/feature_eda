import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

matplotlib.use('TkAgg')
import seaborn as sns
from scipy import stats

import warnings

warnings.filterwarnings("ignore")


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


#   基于模型预测来发现异常数据
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
