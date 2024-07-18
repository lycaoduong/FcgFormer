import torch
import numpy as np
from transformers import AutoModelForImageClassification, AutoConfig


class PredictorCls(object):
    def __init__(self, model_path='lycaoduong/FcgFormer', device='cpu', ckpt=None):
        self.model = AutoModelForImageClassification.from_pretrained(model_path, trust_remote_code=True)
        if ckpt is not None:
            weight = torch.load(ckpt, map_location=device)
            self.model.load_state_dict(weight, strict=True)
        self.model.to(device)
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        self.name_list = list(config.cls_name.keys())
        self.id_list = list(config.cls_name.values())
        self.device = device

    def __call__(self, spectra):
        tensor = self.model.to_pt_tensor(spectra).to(self.device)
        with torch.no_grad():
            o = self.model(tensor)['logits']
        outputs = torch.sigmoid(o).cpu().numpy()
        return outputs[0]

    def get_result(self, result, th=0.5, pos_only=False):
        if pos_only:
            predict_cls = list(np.where(result >= th))[0]
        else:
            result[result < th] = 0.0
            predict_cls = list(np.where(result >= 0))[0]
        fcn_groups = []
        probabilities = []
        for ids in predict_cls:
            position = self.id_list.index(ids)
            cls_name = self.name_list[position]
            prob = result[ids]
            fcn_groups.append(cls_name)
            probabilities.append(prob)
        return fcn_groups, probabilities

    def get_attention(self):
        att = self.model.get_self_attention(layer_value=1)
        att = np.sum(att[0], axis=0)
        return att
