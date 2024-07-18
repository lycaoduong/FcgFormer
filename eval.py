import argparse
from src.evaluator import Evaluation


def get_args():
    parser = argparse.ArgumentParser('Functional Groups Classification Pytorch')
    parser.add_argument('-p', '--project', type=str, default='FcgFormer', help='Project Name')
    parser.add_argument('-m', '--model', type=str, default='lycaoduong/FcgFormer', help='Choosing Model')
    parser.add_argument('-w', '--ckpt', type=str, default=None, help='Loading pretrained weighted, Default from Hugging Face')
    parser.add_argument('-d', '--dataset', type=str, default='FTIR', help='Loading dataset configs file')
    parser.add_argument('-dv', '--device', type=str, default='cuda', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('-bs', '--batch_size', type=int, default=8, help='Init train batch size')
    parser.add_argument('-nw', '--num_worker', type=int, default=8, help='Number of worker for Dataloader')
    parser.add_argument('-th', '--threshold', type=float, default=0.5, help='Threshold')
    parser.add_argument('-pr', '--prc', type=bool, default=True, help='PR Curve Evaluation')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    opt = get_args()
    print("Starting evaluation Model: {}".format(opt.model))
    evaluator = Evaluation(opt)
    evaluator.eval_data_analysis()
    if opt.prc == True:
        evaluator.ROC_PR_Plot(range=[0.05, 0.95, 0.05], fix=0.5)
    evaluator.start()

