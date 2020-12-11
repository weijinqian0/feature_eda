import json


def parse_json(json_str):
    return json.loads(json_str)


def write_json(json_path, object):
    with open(json_path, 'w') as wf:
        wf.write(json.dumps(object))


def read_json(json_path):
    with open(json_path, 'r') as rf:
        line = rf.readline()
        return json.loads(line.strip())


if __name__ == '__main__':
    write_json('./a', ['a', 'b', 'v'])
    print(read_json('./a'))
