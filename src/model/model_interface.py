from abc import ABCMeta, abstractmethod

"""
定义抽象类，实现模型训练训练中所需要的接口
"""


class IModel(metaclass=ABCMeta):

    @abstractmethod
    def data_process(self, data):
        """
        模型相关的数据预处理
        :param maxbytes:
        :return:
        """
        pass

    @abstractmethod
    def fit(self, train_data, label):
        pass

    @abstractmethod
    def predict(self, data, threshold=0.5):
        pass

    @abstractmethod
    def predict_proba(self, data):
        pass

    @abstractmethod
    def metrics(self, data):
        """
        评价模型效果
        :param data:
        :return:
        """
        pass

    @abstractmethod
    def cv_val(self, data):
        """
        交叉验证
        :param data:
        :return:
        """
        pass

    @abstractmethod
    def save_model(self, data):
        """
        评价模型效果
        :param data:
        :return:
        """
        pass

    @abstractmethod
    def load_model(self, data):
        """
        评价模型效果
        :param data:
        :return:
        """
        pass
