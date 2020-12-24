from src.coco.offline_feature import OfflineFeatureHandler
import pandas as pd

from src.feature_handle.base_utils import count_cn_words, count_cn_uniq
from src.feature_handle.feature_generate import list_join_func, list_time_interval_join_func, content_join_func, \
    merge_list


class Feature1(OfflineFeatureHandler):

    def __init__(self, data_path, saved_path, columns):
        super().__init__(data_path, saved_path, columns)

    def get_columns(self):
        return ['uid', 'comment_id', 'time',
                'content_type', 'content', 'comment_score']

    def fill_nan(self):
        self.data = self.data.dropna()

    def feature_filter(self):
        pass

    def feature_select(self):
        pass

    def feature_generate(self):
        self.data = self.feature_generate_1()

    def feature_generate_1(self):
        data = self.data
        # 统计文字个数
        data['cn_count'] = data['content'].apply(count_cn_words)
        data['cn_count_uniq'] = data['content'].apply(count_cn_uniq)
        data = data[data['content'] != '\\N']
        data = data[data['content_type'] != '\\N']
        data = data[data['time'] != '\\N']

        # 看字数的分布
        # paint_dist(data, ['cn_count'])
        agg_dict = {
            'cn_count': list_join_func,
            'cn_count_uniq': list_join_func,
            'comment_id': list_join_func,
            'content_type': list_join_func,
            'time': list_time_interval_join_func,
            'comment_score': list_join_func,
            'content': content_join_func
        }
        rename_dict = {
            'cn_count': 'comment_count_path',
            'cn_count_uniq': 'cn_uniq_path',
            'comment_id': 'comment_id_path',
            'content_type': 'content_type_path',
            'time': 'time_path',
            'comment_score': 'comment_score_path',
            'content': 'comment_path'
        }
        all_data = pd.DataFrame(data['uid'].drop_duplicates())
        print(all_data.shape)
        all_data = merge_list(all_data, 'uid', data, agg_dict, rename_dict)
        self.save_data(all_data, 'generate_1')
        return all_data


if __name__ == '__main__':
    data_path = ''
    save_path = ''
    Feature1(data_path, save_path, ['uid', 'comment_id', 'time',
                                    'content_type', 'content', 'comment_score'])


def get_agg_feature_by_uid():
    """
    uid作为key，获取数据的分类特征
    :return:
    """
    # data = pd.read_csv(data_path, sep='\t',
    #                    encoding='utf-8', error_bad_lines=False)
    # data.columns = ['uid', 'comment_id', 'time',
    #                 'content_type', 'content', 'comment_score']
    # data = data.dropna()
    # 统计文字个数
    # data['cn_count'] = data['content'].apply(count_cn_words)
    # data['cn_count_uniq'] = data['content'].apply(count_cn_uniq)
    # data = data[data['content'] != '\\N']
    # data = data[data['content_type'] != '\\N']
    # data = data[data['time'] != '\\N']

    # print(base_info(data))
    # print(base_describe(data))
    #
    # def content_join_func(x): return "----".join([str(i) for i in x])
    #
    # # 看字数的分布
    # # paint_dist(data, ['cn_count'])
    # agg_dict = {
    #     'cn_count': list_join_func,
    #     'cn_count_uniq': list_join_func,
    #     'comment_id': list_join_func,
    #     'content_type': list_join_func,
    #     'time': list_time_interval_join_func,
    #     'comment_score': list_join_func,
    #     'content': content_join_func
    # }
    # rename_dict = {
    #     'cn_count': 'comment_count_path',
    #     'cn_count_uniq': 'cn_uniq_path',
    #     'comment_id': 'comment_id_path',
    #     'content_type': 'content_type_path',
    #     'time': 'time_path',
    #     'comment_score': 'comment_score_path',
    #     'content': 'comment_path'
    # }
    #
    # print(data.shape)
    # 获取所有不重复的uid
    # all_data = pd.DataFrame(data['uid'].drop_duplicates())
    # print(all_data.shape)
    # all_data = merge_list(all_data, 'uid', data, agg_dict, rename_dict)
    #
    # all_data.to_csv(save_mid_path, sep='\t', index=False)
