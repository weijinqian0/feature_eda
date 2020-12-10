from abc import ABCMeta, abstractmethod

"""
定义抽象类，实现模型训练训练中所需要的接口
"""
class Model(metaclass=ABCMeta):
    @abstractmethod
    def read(self, maxbytes=-1):
        pass

    @abstractmethod
    def write(self, data):
        pass