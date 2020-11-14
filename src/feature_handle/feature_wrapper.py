from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

"""
包装法：按照一定的顺序添加或删除特征，最后选择出特征的集合
前项累加、后项递减、还可以自己定义
后面再补充集中方法
"""


def rfe_wrapper(data, target):
    """
    递归消除的方式进行特征选择
    使用一个基础模型进行多轮训练，每轮训练之后，消除若干权值系数的特征，然后基于新的特征集进行下一轮训练
    可以把阳哥的那个训练步骤搞过来
    :param data:
    :param target:
    :return:
    """
    return RFE(estimator=LogisticRegression(multi_class='auto', solver='lbfgs', max_iter=500),
               n_features_to_select=2).fit_transform(data, data[target])
