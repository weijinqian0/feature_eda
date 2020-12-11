import datetime

import pandas as pd

from sklearn.utils import shuffle
from pandas import DataFrame

from src.coco.online_feature import OnlineFeatureHandler
from src.feature_handle.base_utils import columns_drop, to_csv
from src.model.statistic_model.classfier_model import LrClassifier, KnnClassifier, LgbmClassifier, RFClassifier
from src.util import write_json, read_json


def classifier_model(model_name):
    if model_name == 'lr':
        return LrClassifier()
    elif model_name == 'knn':
        return KnnClassifier()
    elif model_name == 'lgb':
        return LgbmClassifier()
    elif model_name == 'rf':
        return RFClassifier()


class Coco(object):

    def __init__(self, train_path, predict_path, result_path,
                 model_path, label_name, id_name,
                 selected_features, model_name) -> None:
        """
        :param train_path:
        :param predict_path:
        :param result_path:
        :param model_path:
        :param label_name:
        :param id_name:
        :param selected_features: 模型训练之前选择参与训练的数据
        """
        super().__init__()
        self._train_path = train_path
        self._predict_path = predict_path
        self._result_path = result_path
        self._model_path = model_path
        self._label_name = label_name
        self._id_name = id_name
        self._selected_features = selected_features
        self._clf = classifier_model(model_name)
        self.feature_used_name = []
        self.predict_label, self.predict_proba = [], []
        self._model_config = self._model_path + 'model_config'

    def load_train_data(self):
        # 加载数据之后的预处理工作
        train_data = pd.read_csv(self._train_path, sep='\t')
        train_data = shuffle(train_data)
        train_data = train_data.drop(self._id_name, axis=1)
        train_data[self._label_name] = train_data[self._label_name].apply(lambda x: 1 if x > 0.25 else 0)
        print(train_data.columns)
        return train_data

    def load_predict_data(self):
        # 预测数据
        predict_data = pd.read_csv(self._predict_path, sep='\t')
        predict_data = predict_data[get_model_features()]
        print(predict_data.columns)
        return predict_data

    def feature_preprocess(self, train_data):
        """
        提供数据预处理的功能接口，特征清理，特征生成，特征转换，新增特征？
        :param train_data:
        :param predict_data:
        :return:
        """
        handler = OnlineFeatureHandler(label_name=self._label_name, data=train_data)
        handler.pipeline()
        return handler.output()

    def train_and_predict(self, is_save_model=True, threshold=0.5):
        self.train(is_save_model)
        self.predict(use_saved_model=False, threshold=threshold)

    def train(self, is_save_model=True):
        # 加载数据
        train_data = self.load_train_data()
        # 去做特征选择了 返回没有被选择的特征
        X, y, self.feature_used_name = self.feature_preprocess(train_data)
        self._clf.fit(X, y)
        if is_save_model:
            saved_model_path = self._clf.save_model(self._model_path)
            config = {'model_name': saved_model_path, 'features': self.feature_used_name}
            write_json(self._model_config, config)

    def predict(self, use_saved_model=False, threshold=0.5):
        """
        预测并保存预测信息
        :param use_saved_model:
        :param threshold:
        :return:
        """
        if use_saved_model:
            config = read_json(self._model_config)
            saved_model_path, use_features = config['model_name'], config['features']
            self._clf.load_model(saved_model_path)
            self.feature_used_name = use_features

        predict_data = self.load_predict_data()
        # 获取预测数据id列
        user_id = predict_data[self._id_name]
        columns_drop(predict_data, [self._id_name])
        predict_data = predict_data[self.feature_used_name]
        X_test = predict_data.values
        self.predict_label, self.predict_proba = self._clf.predict(X_test)
        self._save_predict_data(user_id, threshold)

    def _save_predict_data(self, uid: DataFrame, threshold):
        """
        保存预测后的数据
        :param uid:
        :param threshold:
        :return:
        """
        if (self.predict_label is None) or (self.predict_proba is None):
            raise BaseException('You should predict firstly.')
        model_info = self._clf.info()
        # 保存概率值
        result_file_name = self._result_path + str(model_info) + '_proba.csv'
        result_data = pd.concat([uid, pd.DataFrame(self.predict_proba)], axis=1)
        to_csv(result_data, result_file_name)

        # 保存label
        result_file_name = self._result_path + str(model_info) + '_proba_to_label_using_th_' + str(threshold) + '.csv'
        result_data = pd.concat([uid, pd.DataFrame(self.predict_label)], axis=1)
        to_csv(result_data, result_file_name)


if __name__ == "__main__":
    base_data_path = '/Users/weijinqian/Documents/feature/feature_eda/data/project'
    result_path = base_data_path + '/model_result/'
    model_path = base_data_path + '/model/'
    train_data_path = base_data_path + '/train_data.txt'
    # 随机整一批用来预测的数据
    predict_data_path = base_data_path + '/predict_data.txt'


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


    coco = Coco(
        train_path=train_data_path,
        predict_path=predict_data_path,
        result_path=result_path,
        model_path=model_path,
        label_name='label',
        id_name='uid',
        selected_features=get_model_features(),
        model_name='lr'

    )
    coco.train(True)
    coco.predict(True)
