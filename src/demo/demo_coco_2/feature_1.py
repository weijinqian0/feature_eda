from src.coco.offline_feature import OfflineFeatureHandler
import pandas as pd

from src.feature_handle.base_data_info import *
from src.feature_handle.base_utils import count_cn_words, count_cn_uniq, read_csv
from src.feature_handle.feature_generate import list_join_func, list_time_interval_join_func, content_join_func, \
    merge_list
from src.feature_handle.view_utils import *

origin_path = '/Users/weijinqian/Documents/feature/feature_eda/data/project/produce_will/base_user_profile_with_tags'


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


data = read_csv(origin_path)
print(data.columns.values.tolist())
base_info(data)
base_describe(data)


def get_view_columns():
    return ['1st_max', '1st_cnt', '2nd_max', '2nd_cnt']


def get_view_columns_1():
    return ['科学教育', '健康生活', '娱乐休闲', '电脑网络', '经济金融', '文化艺术', '企业管理', '社会民生', '体育运动', '电子数码',
            '心理分析', '教育', '生活', '科学技术', '健康知识', ]


data = data[data['1st_max'] <= 80]
data = data[data['2nd_max'] <= 80]
data_pos = data[data['label'] == 1]
data_neg = data[data['label'] == 0]

base_info(data_pos)
base_describe(data_pos)
# paint_dist(data_pos, get_view_columns())
# paint_dist(data_neg, get_view_columns())

paint_dist(data_pos, get_view_columns_1())
paint_dist(data_neg, get_view_columns_1())

# paint_box(data_pos, get_view_columns())
# paint_box(data_neg, get_view_columns())

if __name__ == '__main__':
    pass
