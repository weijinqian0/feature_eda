import pandas as pd

from sklearn.utils import shuffle

from src.base_line.feature import feature_preprocessor
from src.feature_handle.base_utils import columns_drop, to_csv
from src.model.statistic_model.classfier_model import LrClassifier, KnnClassifier, LgbmClassifier, RFClassifier


def get_model_features():
    return ['uid', 'comment_count', 'comment_words_max', 'comment_words_mean',
            'comment_words_min', 'comment_words_std', 'type_ratio', 'video_count',
            'pictext_count', 'other_count', 'time_interval_max',
            'time_interval_mean', 'time_interval_min', 'time_interval_std',
            'comment_score_max', 'comment_score_mean', 'comment_score_min',
            'comment_score_std', 'comment_score_2', 'comment_score_1',
            'comment_score_0', 'percentage_2', 'percentage_1', 'percentage_0',
            'comment_score_2_cross', 'comment_score_1_cross',
            'comment_score_0_cross']


def classifier_model(model_name):
    if model_name == 'lr':
        return LrClassifier()
    elif model_name == 'knn':
        return KnnClassifier()
    elif model_name == 'lgb':
        return LgbmClassifier()
    elif model_name == 'rf':
        return RFClassifier()


def train_model(data_train, data_predict, result_path, model_path, threshold=0.5, label='label',
                is_save_model=True):
    """
    提供一键调用方式
    :param result_path: 结果保存位置
    :param data_train:
    :param data_predict:
    :return:
    """
    data_predict_user_id = data_predict['uid']
    columns_drop(data_predict, ['uid'])
    # 去做特征选择了 这里其实
    X, y, X_test = feature_preprocessor(data_train, data_predict, label)
    # 选择一个模型
    model_name = 'rf'
    model = classifier_model(model_name)

    model.fit(X, y)

    if is_save_model:
        model.save_model(model_path)

    predict_label, predict_proba = model.predict(X_test)

    # 保存预测之后的概率值
    result_file_name = result_path + str(model_name) + '_proba.csv'
    result_data = pd.concat(
        [data_predict_user_id, pd.DataFrame(predict_proba)], axis=1)
    to_csv(result_data, result_file_name)

    # 保存预测结果，卡了阈值之后的结果
    result_file_name = result_path + str(model_name) + '_proba_to_label_using_th_' + str(threshold) + '.csv'
    result_data = pd.concat(
        [data_predict_user_id, pd.DataFrame(predict_label)], axis=1)
    to_csv(result_data, result_file_name)


if __name__ == "__main__":
    base_data_path = '/Users/weijinqian/Documents/feature/feature_eda/data/project'
    result_path = base_data_path + '/model_result/'
    model_path = base_data_path + '/model/'
    train_data_path = base_data_path + '/train_data.txt'
    # 随机整一批用来预测的数据
    predict_data_path = base_data_path + '/predict_data.txt'
    train_data = pd.read_csv(train_data_path, sep='\t')
    train_data = shuffle(train_data)
    train_data = train_data.drop('uid', axis=1)
    train_data['label'] = train_data['label'].apply(lambda x: 1 if x > 0.25 else 0)
    print(train_data.columns)

    # 预测数据
    predict_data = pd.read_csv(predict_data_path, sep='\t')
    predict_data = predict_data[get_model_features()]
    print(predict_data.columns)

    train_model(train_data, predict_data, result_path, model_path=model_path)
