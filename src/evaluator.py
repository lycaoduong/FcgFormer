import datetime
import os
import json
from utils.fcgUtils.trainutils import YamlRead
from utils.fcgUtils import transform as tr
from torchvision import transforms
from utils.fcgUtils.dataloader import FCGClassificationDataset
from torch.utils.data import DataLoader
from transformers import AutoModelForImageClassification, AutoConfig
# from networks.ircharacercnn.ircnn import IrCNN
import torch
from tqdm.autonotebook import tqdm
import traceback
import numpy as np
from utils.fcgUtils.plotutils import func_confusion, subs_confusion, plot_data, plot_conf, plot_roc_pr_curve, subs_len_confusion, plot_numF_Data


class Evaluation(object):
    def __init__(self, eval_opt):
        self.project = eval_opt.project
        self.model_name = eval_opt.model
        self.dataset = eval_opt.dataset

        print('Project Name: ', self.project)
        print('Model and Dataset: ', self.model_name, self.dataset)
        date_time = datetime.datetime.now()
        date_time = date_time.strftime("%Y.%m.%d_%H.%M.%S")
        print('Date Access: ', date_time)


        exp_name, trial_name = 'single_run', date_time
        self.save_dir = './runs/eval/{}/{}/{}/{}/'.format(self.project, self.dataset, exp_name, trial_name)
        os.makedirs(self.save_dir, exist_ok=True)


        # Save train parameters
        with open('{}/eval_params.txt'.format(self.save_dir), 'w') as f:
            json.dump(eval_opt.__dict__, f, indent=2)

        # Read dataset
        dataset_configs = YamlRead(f'configs/dataset/{self.dataset}.yaml')
        self.eval_dir = dataset_configs.test_dir
        self.mean = dataset_configs.mean
        self.std = dataset_configs.std
        self.num_cls = dataset_configs.num_cls

        #Read Model configs
        model_configs = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
        self.signal_size = model_configs.signal_size

        eval_list = []
        for file in os.listdir(self.eval_dir):
            if file.endswith(".npy"):
                label = os.path.join(self.eval_dir, file[:-4] + '.txt')
                if os.path.isfile(label):
                    eval_list.append(file)

        # Data Loader

        self.device = eval_opt.device
        self.batch_size = eval_opt.batch_size


        eval_transforms = [
            tr.Normalizer(with_std=False),
            tr.Resizer(signal_size=self.signal_size)
        ]

        eval_set = FCGClassificationDataset(root_dir=self.eval_dir, list_data=eval_list, max_sequence=None,
                                            transform=transforms.Compose(eval_transforms))

        eval_params = {
            'batch_size': self.batch_size,
            'shuffle': False,
            'drop_last': False,
            'num_workers': eval_opt.num_worker
        }
        self.eval_generator = DataLoader(eval_set, collate_fn=tr.collater, **eval_params)

        # Model
        model = AutoModelForImageClassification.from_pretrained(self.model_name, trust_remote_code=True)

        # if torch.cuda.is_available():
        #     model = nn.DataParallel(model)
        if eval_opt.ckpt is not None:
            weight = torch.load(eval_opt.ckpt, map_location=self.device)
            model.load_state_dict(weight, strict=True)

        self.model = model
        self.model = self.model.to(self.device)

        self.num_iter_per_epoch = len(self.eval_generator)
        self.step = 0

        self.funcs_confusion = np.zeros((self.num_cls, 2, 2))
        self.substance_confusion = np.zeros((8, 2))
        self.cls_dic = dataset_configs.pos_dic
        self.data_dis = np.zeros((1, self.num_cls))
        self.subs_dis = np.zeros((1, 7))
        self.cls_weight = np.zeros((1, self.num_cls))
        self.loss_weight = np.zeros((1, self.num_cls))
        self.th = eval_opt.threshold
        self.best_th = eval_opt.threshold
        self.true_subs = []
        
    def eval_data_analysis(self):
        progress_bar = tqdm(self.eval_generator)
        for iter, data in enumerate(progress_bar):
            signals, label = data['signal'], data['label']
            label = label.cpu().numpy()
            self.data_dis += np.sum(label, axis=0)
            for lb in label:
                len = np.sum(lb)
                self.subs_dis[:, len-1] += 1

        sum = np.sum(self.data_dis, axis=1)
        self.cls_weight = self.data_dis / sum
        self.loss_weight = (1.0 / self.cls_weight)
        plot_data(data_dis=self.data_dis, save_dir=self.save_dir, save_name="data_distribution.png")
        plot_numF_Data(data_dis=self.subs_dis, save_dir=self.save_dir, save_name="numF_distribution.png")

    def plot_confusion_matrix(self):
        # Plot total functional groups cf
        plot_conf(np.sum(self.funcs_confusion, axis=0), labelX=["Positive", "Negative"],
                  labelY=["Positive", "Negative"],
                  title="Total Functional groups Confusion Matrix",
                  save_dir=self.save_dir, save_name="fngs_cf.png", rotationY=90)
        # Plot subs cf
        # tem = np.zeros((2, 2))
        # tem[0] = self.substance_confusion
        plot_conf(self.substance_confusion, labelX=["True", "False"],
                  labelY=["1-Group", "2-Group", "3-Group", "4-Group", "5-Group", "6-Group", "7-Group", "Total"],
                  title="Molecule Confusion Matrix", save_dir=self.save_dir,
                  size=(19, 12),
                  save_name="subs_cf.png", rotationY=0)

        # Plot each functional group cf
        for i in self.cls_dic.keys():
            fng_name = self.cls_dic[i]
            conf = self.funcs_confusion[i]
            plot_conf(conf, labelX=["Positive", "Negative"],
                      labelY=["Positive", "Negative"],
                      title="{} Confusion Matrix".format(fng_name.capitalize()),
                      save_dir=self.save_dir,
                      save_name="{}_cf.png".format(fng_name), rotationY=90)

        print("Finish plot Confusion matrix, check save path: {}".format(self.save_dir))

    def PR_Calculator(self):
        total_func_confusion = np.sum(self.funcs_confusion, axis=0)
        ext = 1e-5
        tp = total_func_confusion[0, 0]
        fn = total_func_confusion[0, 1]
        fp = total_func_confusion[1, 0]
        tn = total_func_confusion[1, 1]
        precision = tp / (tp + fp + ext)
        recall = tp / (tp + fn + ext) #TPR
        fpr = fp / (fp + tn + ext)
        return precision, recall, fpr

    def ROC_PR_Plot(self, range=[0.05, 0.95, 0.05], fix=None):
        th_array = np.arange(start=range[0], stop=range[1] + range[2], step=range[2])
        precision = []
        recall = []
        fp = []
        for th in th_array:
            print("PR Evaluation - threshold: {}".format(th))
            self.th = th
            self.eval()
            p, r, fpr = self.PR_Calculator()
            print("Precision-{}; Recall-{}; FPR-{}".format(p, r, fpr))
            recall.append(r)
            precision.append(p)
            fp.append(fpr)
            # Reset value
            self.funcs_confusion = np.zeros((self.num_cls, 2, 2))
            self.substance_confusion = np.zeros((8, 2))
        optimal_idx = np.argmax(np.array(recall)-np.array(fp))
        if fix is None:
            optimal_threshold = th_array[optimal_idx]
        else:
            optimal_threshold=fix
        self.th = optimal_threshold
        print("Optimal Threshold value is:", optimal_threshold)
        plot_roc_pr_curve(precision, recall, fp, save_dir=self.save_dir)

    def eval(self):
        self.model.eval()
        last_epoch = self.step // self.num_iter_per_epoch
        progress_bar = tqdm(self.eval_generator)
        losses = []

        for iter, data in enumerate(progress_bar):
            if iter < self.step - last_epoch * self.num_iter_per_epoch:
                progress_bar.update()
                continue
            try:
                signals, label, fn = data['signal'], data['label'], data['fn']
                signals = signals.to(self.device)
                label = label.to(self.device)
                label = label.to(torch.float32)

                with torch.no_grad():
                    loss, predict = self.model(signals, label)['loss'], self.model(signals, label)['logits']


                fun_conf = func_confusion(target=label.cpu().numpy(), result=predict.cpu().numpy(), th=self.th)
                self.funcs_confusion += fun_conf

                # subs_conf = subs_confusion(target=label.cpu().numpy(), result=predict.cpu().numpy(), th=self.th)
                subs_conf, list_true = subs_len_confusion(target=label.cpu().numpy(), result=predict.cpu().numpy(),
                                                         th=self.th, fn=fn)

                self.true_subs.extend(list_true)
                self.substance_confusion += subs_conf

                losses.append(float(loss.item()))

                descriptor = '[Eval] Step: {}. Iteration: {}/{}. Loss: {:.6f}.'.format(
                        self.step, iter + 1, self.num_iter_per_epoch, loss.item())
                progress_bar.set_description(descriptor)
                self.step += 1

            except Exception as e:
                print('[Error]', traceback.format_exc())
                print(e)
                continue

        mean_loss = np.mean(losses)
        eval_descrip = '[Eval]. Mean Loss: {:.6f}.'.format(mean_loss)
        print(eval_descrip)

    def start(self):
        # self.eval_data_analysis()
        self.eval()
        # print(self.true_subs)
        self.plot_confusion_matrix()
