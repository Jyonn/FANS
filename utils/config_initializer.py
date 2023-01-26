import os
import re

import yaml
from oba import Obj

from utils.smart_printer import printer, Color, Bracket


class ConfigInitializer:
    print = printer[('CONF-INIT', Bracket.CLASS, Color.MAGENTA)]

    @staticmethod
    def get_config_value(config: Obj, path: str):
        path = path.split('.')
        path_ = []
        for key in path:
            list_keys = key.split('[')
            for list_key in list_keys:
                if list_key.endswith(']'):
                    list_key = int(list_key[:-1])
                path_.append(list_key)

        value = config
        for key in path_:
            value = value[key]
        return value

    @classmethod
    def format_config_path(cls, config: Obj, path: str):
        dynamic_values = re.findall('{.*?}', path)
        for dynamic_value in dynamic_values:
            path = path.replace(dynamic_value, str(cls.get_config_value(config, dynamic_value[1:-1])))
        return path

    @classmethod
    def init(cls, config_path, exp_path):
        config = yaml.safe_load(open(config_path))
        config = Obj(config)

        exp = yaml.safe_load(open(exp_path))
        exp = Obj(exp)

        exp.model = exp.model.upper()
        cls.print('model:', exp.model)

        meta_config = Obj(dict(exp=exp, config=config))

        if config.store.data_dir:
            config.store.data_dir = cls.format_config_path(meta_config, config.store.data_dir)
        config.store.save_dir = cls.format_config_path(meta_config, config.store.save_dir)
        exp.exp = cls.format_config_path(meta_config, exp.exp)

        if exp.load.load_ckpt:
            exp.load.load_ckpt = cls.format_config_path(meta_config, exp.load.load_ckpt)
        if exp.load.ckpt_base_path:
            exp.load.ckpt_base_path = cls.format_config_path(meta_config, exp.load.ckpt_base_path)

        config.store.ckpt_path = os.path.join(config.store.save_dir, exp.exp)
        config.store.log_path = os.path.join(config.store.ckpt_path, '{}.log'.format(exp.exp))

        os.makedirs(config.store.ckpt_path, exist_ok=True)

        return config, exp
