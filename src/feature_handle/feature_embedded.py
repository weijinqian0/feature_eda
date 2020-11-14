from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression

"""
特征嵌入的方式
类型：
1. 基于惩罚项的特征选择；
2. 基于树模型的特征选择

"""


def penalty_embedded(data, target):
    """
    将带有L1惩罚项的逻辑回归作为基模型的特征选择
    L1惩罚项降维的原理在于保留多个对目标值具有同等相关性的特征中的一个，所以没选到的特征不代表不重要。故，可结合L2惩罚项来优化。
    具体操作为：若一个特征在L1中的权值为1，选择在L2中权值差别不大且在L1中权值为0的特征构成同类集合，将这一集合中的特征平分L1中的权值，
    故需要构建一个新的逻辑回归模型
    :param data:
    :param target:
    :return:
    """
    return SelectFromModel(LogisticRegression(penalty="l1", C=0.1)).fit_transform(data, data[target])


def tree_embedded(data, target):
    """
    使用树模型进行特征选择
    :param data:
    :param target:
    :return:
    """
    return SelectFromModel(GradientBoostingClassifier()).fit_transform(data, data[target])
