import json
import re
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


if __name__ == '__main__':
    print(get_numbers("abc123"))
    # write_json('./a', ['a', 'b', 'v'])
    # print(read_json('./a'))
