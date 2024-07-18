import argparse
from src.measurer import Measurer


def get_args():
    parser = argparse.ArgumentParser('Functional Groups Classification Pytorch')
    parser.add_argument('-p', '--project', type=str, default='FcgFormer', help='Project Name')
    parser.add_argument('-m', '--model', type=str, default='lycaoduong/FcgFormer', help='Choosing Model')
    parser.add_argument('-w', '--ckpt', type=str, default=None, help='Loading pretrained weighted')
    parser.add_argument('-d', '--dataset', type=str, default='FTIR', help='Loading pretrained weighted, Default from Hugging Face')
    parser.add_argument('-dv', '--device', type=str, default='cuda', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    opt = get_args()
    print("Starting measure Model: {}".format(opt.model))
    measurer = Measurer(opt)
    measurer.start()
