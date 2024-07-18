import yaml
import numpy as np
import torch


class YamlRead:
    def __init__(self, params_path):
        self.params = yaml.safe_load(open(params_path, encoding='utf-8').read())

    def update(self, dictionary):
        self.params = dictionary

    def __getattr__(self, item):
        return self.params.get(item, None)


class toOpt(object):
    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float(1e5)

    def __call__(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False