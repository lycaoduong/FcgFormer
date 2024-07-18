import argparse
from src.FcgTrainer import Trainer


def get_args():
    parser = argparse.ArgumentParser('Functional Groups Classification Pytorch')
    parser.add_argument('-p', '--project', type=str, default='FcgFormer', help='Project Name')
    parser.add_argument('-m', '--model', type=str, default='lycaoduong/FcgFormer', help='Choosing Model')
    parser.add_argument('-w', '--ckpt', type=str, default=None, help='Loading pretrained weighted, Default from Hugging Face')
    parser.add_argument('-d', '--dataset', type=str, default='FTIR', help='Loading dataset configs file')
    parser.add_argument('-au', '--aug', type=str, default='aug', help='Loading Augmentation configs file')
    parser.add_argument('-lr', '--lr', type=float, default=2e-4, help='Init Learning Rate')
    parser.add_argument('-ep', '--epochs', type=int, default=100, help='Init number of train epochs')
    parser.add_argument('-dv', '--device', type=str, default='cuda', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('-bs', '--batch_size', type=int, default=4, help='Init train batch size')
    parser.add_argument('-nw', '--num_worker', type=int, default=8, help='Number of worker for Dataloader')
    parser.add_argument('-op', '--optimizer', type=str, default='adamw', help='Choosing optimizer')
    parser.add_argument('-ls', '--lr_scheduler', type=str, default='cosine', help='Choosing learning rate scheduler')
    args = parser.parse_args()
    # args = vars(parser.parse_args())
    return args


if __name__ == '__main__':
    opt = get_args()
    trainer = Trainer(opt)
    # trainer.data_analysis()
    trainer.start()
