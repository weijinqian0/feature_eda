import json
import re
import time

import numpy as np


def parse_json(json_str):
    return json.loads(json_str)


def write_json(json_path, object):
    with open(json_path, 'w') as wf:
        wf.write(json.dumps(object))


def read_json(json_path):
    with open(json_path, 'r') as rf:
        line = rf.readline()
        return json.loads(line.strip())


def get_numbers(text):
    return re.sub("\D", '', text)


def round(value):
    return np.round(value, 3)


def char_count(text, anchor):
    """
    字符在文本中出现的次数
    :param text:
    :param anchor:
    :return:
    """
    return len(text.split(anchor)) - 1


def date_to_stamp(date):
    """将日期转化为时间戳
    :param date: 待转化的日期
    :return: 转化后的时间数据
    """
    # 先转换为时间数组，完整的为"%Y-%m-%d %H:%M:%S"
    timeArray = time.strptime(date, "%Y-%m-%d")
    # 转换为时间戳
    time_stamp = int(time.mktime(timeArray))

    return time_stamp


def stamp_to_time(time_stamp):
    """将时间戳转化成普通时间的格式
    :param time_stamp: 时间戳
    :return: 时间戳对应的日期
    """
    stamp = time.localtime(time_stamp)
    local_time = time.strftime("%Y-%m-%d", stamp)

    return local_time


if __name__ == '__main__':
    print(get_numbers("abc123"))
    print(char_count('我们是谁是谁是啥', '是'))
    # write_json('./a', ['a', 'b', 'v'])
    # print(read_json('./a'))
