import pandas as pd

from src.feature_handle.base_utils import read_csv
from src.feature_handle.view_utils import paint_box, paint_dist, paint_violin, paint_dist_single

data_path = '/Users/weijinqian/Documents/feature/feature_eda/data/project/produce_with_reply_2_year_no_import_with_browse'

data = read_csv(data_path, columns=['uid', 'reply'])
# data['reply'] = data[data['reply'] <= 500]

# paint_dist_single(data, ['reply'], bins=[0, 50, 100, 500, 1000])

print(data.shape)
print(data[data['reply'] > 1000].shape)
print(data[(data['reply'] <= 1000) & (data['reply'] > 500)].shape)
print(data[(data['reply'] <= 500) & (data['reply'] > 100)].shape)
print(data[(data['reply'] <= 100) & (data['reply'] > 80)].shape)
print(data[(data['reply'] <= 80) & (data['reply'] > 50)].shape)
print(data[(data['reply'] <= 50) & (data['reply'] > 30)].shape)
print(data[(data['reply'] <= 30) & (data['reply'] > 10)].shape)
print(data[(data['reply'] <= 10) & (data['reply'] > 5)].shape)
print(data[(data['reply'] <= 5)].shape)

