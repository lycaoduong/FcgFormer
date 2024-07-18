import numpy as np
from src.predictor import PredictorCls
import argparse
from utils.fcgUtils.plotutils import plot_self_attention_map


def get_args():
    parser = argparse.ArgumentParser('Functional Groups Classification Pytorch')
    parser.add_argument('-p', '--project', type=str, default='FcgFormer', help='Project Name')
    parser.add_argument('-m', '--model', type=str, default='lycaoduong/FcgFormer', help='Choosing Model')
    parser.add_argument('-w', '--ckpt', type=str, default=None, help='Loading pretrained weighted, Default from Hugging Face')
    parser.add_argument('-dv', '--device', type=str, default='cuda', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('-op', '--input', type=str, default='./dataset/test.npy', help='Choosing input')
    parser.add_argument('-th', '--threshold', type=float, default=0.5, help='Choosing threshold')
    parser.add_argument('-att', '--plotAtt', type=bool, default=False, help='Plot Attention Map')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    opt = get_args()
    engine = PredictorCls(model_path=opt.model, device=opt.device, ckpt=opt.ckpt)

    spectra = np.load(opt.input)
    outputs = engine(spectra)
    fcn_groups, probabilities = engine.get_result(outputs, th=opt.threshold, pos_only=True)
    print(fcn_groups)
    print(probabilities)
    if opt.plotAtt:
        att = engine.get_attention()
        plot_self_attention_map(spectra, att)
