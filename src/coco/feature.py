from pandas import DataFrame

from src.feature_handle.base_data_info import base_info, base_describe
from src.feature_handle.feature_embedded import lgb_embeded_1


def fill_nan(train_data: DataFrame, label_name):
    """
    缺失值处理
    :param train_data:
    :param label_name:
    :return:
    """
    train_target = train_data[label_name]
    train_data = train_data.drop(label_name, axis=1)
    train_data.fillna(value=train_data.median())

    return train_data, train_target


def feature_select(name, train_data: DataFrame, target, feature_name):
    if name == "lgb":
        return lgb_embeded_1(train_data, target, feature_name, '0.1*mean')
    else:
        return train_data, target, feature_name, []


def feature_preprocessor(data_train, label_name):
    """
    数据预处理
    :return:
    """

    print(base_info(data_train))
    print(base_describe(data_train))

    # 填充nan
    train_data, train_target = fill_nan(data_train, label_name)

    X = train_data.values
    y = train_target.values
    # 特征选择
    X, feature_score_dict_sorted, feature_used_name, feature_not_used_name = feature_select(
        '', X, y, train_data.columns.tolist())

    print("当前选中的特征维度" + str(len(feature_used_name)))
    print(feature_used_name)
    print(feature_not_used_name)
    return X, y, feature_used_name
