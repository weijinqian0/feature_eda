from src.feature_handle.feature_generate import *


def get_merchant_feature(feature: DataFrame):
    merchant = feature[['merchant_id', 'coupon_id', 'distance', 'data_received', 'date']].copy()
    t = merchant[['merchant_id']].copy()
    # 删除重复行数据
    t.drop_duplicates(inplace=True)

    # 每个商户的交易总次数
    t1 = merchant[merchant.date != 'null'][['merchant_id']].copy()
    merchant_feature = add_count_new_feature(t, t1, 'merchant_id', 'total_sales')

    # 在每个商户销售中，使用优惠券的交易次数（正样本）
    t2 = merchant[(merchant.date != 'null') & (merchant.coupon_id != 'null')][['merchant_id']].copy()
    merchant_feature = add_count_new_feature(t, t2, 'merchant_id', 'sales_use_coupon')

    # 每个商户发放的优惠券总数
    t3 = merchant[merchant.coupon_id != 'null'][['merchant_id']].copy()
    merchant_feature = add_count_new_feature(merchant_feature, t3, 'merchant_id', 'total_coupon')

    # 在每个线下商户含有优惠券的交易中，统计与用户距离的最大值、最小值、平均值、中位值
    t4 = merchant[(merchant.date != 'null') & (merchant.coupon_id != 'null') & (merchant.distance != 'null')][
        ['merchant_id', 'distance']].copy()

    t4.distance = t4.distance.astype('int')
    merchant_feature = add_agg_feature(merchant_feature, t4, ['merchant_id'], 'distance',
                                       ['min', 'max', 'mean', 'median'], 'merchant')
    merchant_feature.sales_use_coupon = merchant_feature.sales_use_coupon.replace(np.nan, 0)

    # 商户发放的优惠券的使用率
    merchant_feature['merchant_coupon_transfer_rate'] = merchant_feature.sales_use_coupon.astype(
        'float') / merchant_feature.total_coupon
    # 在商户交易中，使用优惠券的交易占比
    merchant_feature['coupon_rate'] = merchant_feature.sales_use_coupon.astype('float') / merchant_feature.total_sales

    merchant_feature.total_coupon = merchant_feature.total_coupon.replace(np.nan, 0)
    return merchant_feature


def add_day_gap(t10):
    pass


def is_firstlastone(t10):
    pass


def get_day_gap_before():
    pass


def get_day_gap_after():
    pass


def get_user_feature(feature: DataFrame):
    user = feature[['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received', 'date']].copy()
    t = user[['user_id']].copy()
    t.drop_duplicates(inplace=True)

    # 每个用户交易的商户数
    t1 = user[user.date != 'null'][['user_id', 'merchant_id']].copy()
    t1.drop_duplicates(inplace=True)
    t1 = t1[['user_id']]
    user_feature = add_count_new_feature(t, t1, 'user_id', 'count_merchant')

    # 在每个用户现象使用优惠券产生的交易中，统计和商户距离的最大值、最小值、平均值、中位值
    t2 = user[(user.date != 'null') & (user.coupon_id != 'null') & (user.distance != 'null')][
        ['user_id', 'merchant_id']]
    t2.distance = t2.distance.astype('int')
    user_feature = add_agg_feature(t, t2, ['user_id'], 'distance', ['min', 'max', 'mean', 'median'], 'user')

    # 每个用户使用优惠券消费的次数
    t7 = user[(user.date != 'null') & (user.coupon_id != 'null')][['user_id']]
    user_feature = add_count_new_feature(user_feature, t7, 'user_id', 'buy_use_coupon')

    # 每个用户消费的总次数
    t8 = user[(user.date != 'null')][['user_id']]
    user_feature = add_count_new_feature(user_feature, t8, 'user_id', 'buy_total')

    # 每个用户收到优惠券的总数
    t9 = user[(user.date != 'null')][['user_id']]
    user_feature = add_count_new_feature(user_feature, t9, 'user_id', 'coupon_received')

    # 用户从收到优惠券到用券消费的时间间隔，统计其最大值、最小值、平均值、中位值
    t10 = user[(user.date_received != 'null') & (user.date != 'null')][['user_id', 'date_received', 'date']]
    t10 = add_day_gap(t10)
    t10 = t10[['user_id', 'day_gap']]
    user_feature = add_agg_feature(user_feature, t10, ['user_id'], 'day_gap', ['min', 'max', 'mean', 'median'], 'user')

    user_feature.count_merchant = user_feature.count_merchant.replace(np.nan, 0)
    user_feature.buy_use_coupon = user_feature.buy_use_coupon.replace(np.nan, 0)

    # 统计用户用券消费在总消费中的占比
    user_feature['bug_use_coupon_rate'] = user_feature.buy_use_coupon.astype('float') / user_feature.buy_total.astype(
        'float')

    user_feature['user_coupon_transfer_rate'] = user_feature.buy_use_coupon.astype(
        'float') / user_feature.coupon_received.astype('float')

    user_feature.buy_total = user_feature.buy_total.replace(np.nan, 0)
    user_feature.coupon_received = user_feature.coupon_received.astype('float')

    # 将数据中的Nan用0替换
    user_feature.buy_total = user_feature.buy_total.replace(np.nan, 0)
    user_feature.coupon_received = user_feature.coupon_received.replace(np.nan, 0)
    return user_feature


def get_user_merchant_feature(feature: DataFrame):
    """
    用户和商户关系：一个用户和一个商家的关系
    :param feature:
    :return:
    """
    t = feature[['user_id', 'merchant_id']].copy()
    t.drop_duplicates(inplace=True)
    # 一个用户在一个商家交易的总次数
    t0 = feature[['user_id', 'merchant_id', 'date']].copy()
    t0 = t0[t0.date != 'null'][['user_id', 'merchant_id']]
    user_merchant = add_count_new_feature(t, t0, ['user_id', 'merchant_id'], 'user_merchant_buy_total')

    # 一个用户在一个商家一共收到的优惠券数量
    t1 = feature[['user_id', 'merchant_id', 'coupon_id']]
    t1 = t1[t1.coupon_id != 'null'][['user_id', 'merchant_id']]
    user_merchant = add_count_new_feature(user_merchant, t1, ['user_id', 'merchant_id'], 'user_merchant_received')

    # 一个用户在一个商家使用优惠券消费的次数
    t2 = feature[['user_id', 'merchant_id', 'date', 'date_received']]
    t2 = t2[(t2.date != 'null') & (t2.date_received != 'null')][['user_id', 'merchant_id']]
    user_merchant = add_count_new_feature(user_merchant, t2, ['user_id', 'merchant_id'], 'user_merchant_buy_use_coupon')

    # 一个用户在一个商家的到店次数
    t3 = feature[['user_id', 'merchant_id']]
    user_merchant = add_count_new_feature(user_merchant, t3, ['user_id', 'merchant_id'], 'user_merchant_any')

    # 一个用户在一个商家没有使用优惠券消费的次数
    t4 = feature[['user_id', 'merchant_id', 'date', 'coupon_id']]
    t4 = t4[(t4.date != 'null') & (t4.coupon_id != 'null')][['user_id', 'merchant_id']]
    user_merchant = add_count_new_feature(user_merchant, t4, ['user_id', 'merchant_id'], 'user_merchant_buy_common')

    user_merchant.user_merchant_buy_use_coupon = user_merchant.user_merchant_buy_use_coupon.replace(np.nan, 0)
    user_merchant.user_merchant_buy_common = user_merchant.user_merchant_buy_common.replace(np.nan, 0)

    # 一个用户对一个商家发放优惠券的使用率
    user_merchant['user_merchant_coupon_transfer_rate'] = user_merchant.user_merchant_buy_use_coupon.astype(
        'float') / user_merchant.user_merchant_received.astype('float')
    # 一个用户在一个商家的总消费次数中，使用优惠券消费的占比
    user_merchant['user_merchant_coupon_buy_rate'] = user_merchant.user_merchant_buy_use_coupon.astype(
        'float') / user_merchant.user_merchant_received.astype('float')

    # 一个用户到点消费的可能性统计
    user_merchant['user_merchant_rate'] = user_merchant.user_merchant_buy_total.astype(
        'float') / user_merchant.user_merchant_any.astype('float')
    # 一个用户在一个商家总消费次数中，不用优惠券的消费次数占比
    user_merchant['user_merchant_common_buy_rate'] = user_merchant.user_merchant_buy_common.astype(
        'float') / user_merchant.user_merchant_buy_total.astype('float')

    return user_merchant


def get_leakage_feature(dataset: DataFrame):
    """
    Leakage 特征群，就是会出现信息泄露的特征
    :param dataset:
    :return:
    """
    t = dataset[['user_id']].copy()
    t['this_month_user_receive_all_coupon_count'] = 1
    t = t.groupby('user_id').agg('sum').reset_index()

    t1 = dataset[['user_id', 'coupon_id']].copy()
    t1['this_month_user_receive_same_coupon_count'] = 1

    t1 = t1.groupby(['user_id', 'coupon_id', 'date_received']).agg('sum').reset_index()
    t2 = dataset[['user_id', 'coupon_id', 'date_received']].copy()
    # 如果出现相同的用户接受相同的优惠券，则在时间上用'：'连接上第n次接受优惠券的时间
    t2 = t2.groupby(['user_id', 'merchant_id'])['date_received'].agg(lambda x: ':'.join(x)).reset_index()

    # 将接收时间的一组按'：'分开，这样就可以计算所接收优惠券的数量
    # apply是合并
    t2['receive_number'] = t2.date_received.apply(lambda s: len(s.split(':')))
    # 过滤出来了有过相似行为的用户
    t2 = t2[t2.receive_number > 1]
    # 最大接收的日期
    t2['max_date_received'] = t2.date_received.apply(lambda s: max([int(d) for d in s.split(':')]))
    # 最小的接收日期
    t2['min_date_received'] = t2.date_received.apply(lambda s: min([int(d) for d in s.split(':')]))
    t2 = t2[['user_id', 'coupon_id', 'max_date_received', 'min_date_received']]

    t3 = dataset[['user_id', 'coupon_id', 'date_received']]
    # 将两个表融合只保留左表数据，相当于保留了最近接收时间和最远接收时间
    t3 = pd.merge(t3, t2, on=['user_id', 'coupon_id'], how='left')
    # 这个优惠券最近接收时间
    t3['this_month_user_receive_same_coupon_lastone'] = t3.max_date_received - t3.data_received.astype(int)
    # 这个优惠券最远接收时间
    t3['this_month_user_receive_same_coupon_firstone'] = t3.data_received.astype(int) - t3.min_date_received

    t3.this_month_user_receive_same_coupon_lastone = t3.this_month_user_receive_same_coupon_lastone.apply(
        is_firstlastone)
    t3 = t3[['user_id', 'coupon_id', 'date_received', 'this_month_user_receive_same_coupon_lastone',
             'this_month_user_receive_same_coupon_firstone']]

    # 提取第四个特征，一个用户所接收到的所有优惠券的数量
    t4 = dataset[['user_id', 'date_received']].copy()
    t4['this_day_receive_all_coupon_count'] = 1
    t4 = t4.groupby(['user_id', 'date_received']).agg('sum').reset_index()

    # 提取第五个特征，一个用户不同时间所接收到的不同优惠券的数量
    t5 = dataset[['user_id', 'coupon_id', 'date_received']].copy()
    t5['this_day_user_receive_same_coupon_count'] = 1
    t5 = t5.groupby(['user_id', 'coupon_id', 'date_received']).agg('sum').reset_index()

    # 一个用户不同优惠券的接收时间
    t6 = dataset[['user_id', 'coupon_id', 'date_received']].copy()
    t6.date_received = t6.date_received.astype('str')
    t6 = t6.groupby(['user_id', 'coupon_id']).agg(lambda x: ':'.join(x)).reset_index()
    t6.rename(columns={'date_received': 'dates'}, inplace=True)

    t7 = dataset[['user_id', 'coupon_id', 'date_received']]
    t7 = pd.merge(t7, t6, on=['user_id', 'coupon_id'], how='left')
    t7['date_received_date'] = t7.date_received.astype('str') + '-' + t7.dates
    t7['day_gap_before'] = t7.date_received_date.apply(get_day_gap_before)
    t7['day_gap_after'] = t7.date_received_date.apply(get_day_gap_after)
    t7 = t7[['user_id', 'coupon_id', 'date_received', 'day_gap_before', 'day_gap_after']]

    other_feature = pd.merge(t1, t, on='user_id')
    other_feature = pd.merge(other_feature, t3, on=['user_id', 'coupon_id'])
    other_feature = pd.merge(other_feature, t4, on=['user_id', 'date_received'])
    other_feature = pd.merge(other_feature, t5, on=['user_id', 'coupon_id', 'date_received'])
    other_feature = pd.merge(other_feature, t7, on=['user_id', 'coupon_id', 'date_received'])
    return other_feature


def add_discount(dataset):
    return pd.DataFrame()


def add_label(result):
    return pd.DataFrame()


def f1(dataset, if_train):
    result = add_discount(dataset)
    result.drop_duplicates(inplace=True)
    if if_train:
        result = add_label(result)
    return result


def f2(dataset, if_train):
    result = add_discount(dataset)
    result.drop_duplicates(inplace=True)
    merchant_feature = get_merchant_feature(dataset)
    result = result.merge(merchant_feature, on='merchant_id', how='left')
    user_feature = get_user_feature(dataset)
    result = result.merge(user_feature, on='user_id', how='left')
    user_merchant = get_user_merchant_feature(dataset)
    result = result.merge(user_merchant, on=['user_id', 'merchant_id'], how='left')
    result.drop_duplicates(inplace=True)
    if if_train:
        result = add_label(result)
    return result


def f3(dataset, if_train):
    result = add_discount(dataset)
    merchant_feature = get_merchant_feature(dataset)
    result = result.merge(merchant_feature, on='merchant_id', how='left')
    user_feature = get_user_feature(dataset)
    result = result.merge(user_feature, on='user_id', how='left')
    user_merchant = get_user_merchant_feature(dataset)
    result = result.merge(user_merchant, on=['user_id', 'merchant_id'], how='left')
    leak_feature = get_leakage_feature(dataset)
    result = result.merge(leak_feature, on=['user_id', 'coupon_id', 'date_received'], how='left')
    result.drop_duplicates(inplace=True)
    if if_train:
        result = add_label(result)
    return result


data_path = ''
feature_path = ''


def normal_feature_generate(feature_function):
    off_train = pd.read_csv(data_path + 'ccf_offline_stagel_train.csv', header=0, keep_default_na=False)
    off_train.columns = ['user_id', 'merchant_id', 'coupon_id', 'discount_id', 'distance', 'date_received', 'date']

    off_test = pd.read_csv(data_path + 'ccf_offline_stagel_test-revised.csv', header=0, keep_default_na=False)
    off_test.columns = ['user_id', 'merchant_id', 'coupon_id', 'discount_id', 'distance', 'date_received', 'date']

    off_train = off_train[
        (off_train.coupon_id != 'null') & (off_train.date_received != 'null') & (off_train.date_received >= '20160501')]

    dftrain = feature_function(off_train, True)
    dftest = feature_function(off_test, False)

    dftrain.drop(['date'], axis=1, inplace=True)
    dftrain.drop(['merchant_id'], axis=1, inplace=True)
    dftest.drop(['merchant_id'], axis=1, inplace=True)

    dftrain.to_csv(feature_path + 'train_' + feature_function.__name__ + '.csv', index=False, sep=',')
    dftest.to_csv(feature_path + 'test_' + feature_function.__name__ + '.csv', index=False, sep=',')
