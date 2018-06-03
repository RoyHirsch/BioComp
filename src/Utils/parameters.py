import json


def parameters(config_path):
    try:
        with open(config_path) as config_json:
            config = json.load(config_json)
            config_json.close()
        return config
    except NameError as ex:
        print("Read Error: no file named %s" % config_path)
        raise ex
