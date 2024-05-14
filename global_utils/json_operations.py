import json


def dict_to_dict(_dict: dict):
    result = {}
    for k, v in _dict.items():
        result[k] = v.to_dict()
    return result


def list_to_dict(_list: dict):
    result = []
    for itm in _list:
        result.append(itm.to_dict())
    return result


def write_json_to_file(json_dict, file_path):
    with open(file_path, 'w') as json_file:
        json.dump(json_dict, json_file)


def read_json_to_dict(model_store_dict):
    with open(model_store_dict, 'r') as file:
        data = json.load(file)
    return data
