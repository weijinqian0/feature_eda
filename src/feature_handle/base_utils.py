"""
特征处理过程中常用的函数
"""


def count_cn_words(data):
    """
    获取中文字个数
    """
    if not isinstance(data, str):
        return 0
    count = 0
    for s in data:
        # 中文字符范围
        if u'\u4e00' <= s <= u'\u9fff':
            count += 1
    return count
