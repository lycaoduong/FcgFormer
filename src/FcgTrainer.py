import datetime
import os
import json
from utils.fcgUtils.trainutils import YamlRead
from utils.fcgUtils import transform as tr
from torchvision import transforms
from utils.fcgUtils.dataloader import FCGClassificationDataset
from torch.utils.data import DataLoader
from transformers import AutoModelForImageClassification, AutoConfig
import torch
from tqdm.autonotebook import tqdm
import traceback
from tensorboardX import SummaryWriter
import numpy as np
from sklearn.model_selection import train_test_split
import nni
from utils.fcgUtils.plotutils import plot_data
from utils.fcgUtils.trainutils import EarlyStopper


class Trainer(object):
    def __init__(self, train_opt):
        self.project = train_opt.project
        self.model_name = train_opt.model
        self.dataset = train_opt.dataset

        print('Project Name: ', self.project)
        print('Model and Dataset: ', self.model_name, self.dataset)
        date_time = datetime.datetime.now()
        date_time = date_time.strftime("%Y.%m.%d_%H.%M.%S")
        print('Date Access: ', date_time)

        if hasattr(train_opt, 'nni'):
            exp_name, _, trial_name = os.environ['NNI_OUTPUT_DIR'].split('\\')[-3:]
            self.nni_writer = True
            self.save_dir = '../runs/nni/{}/{}/{}/{}/'.format(self.project, self.dataset, exp_name, trial_name)
        else:
            exp_name, trial_name = 'single_run', date_time
            self.nni_writer = False
            self.save_dir = './runs/train/{}/{}/{}/{}/'.format(self.project, self.dataset, exp_name, trial_name)
        os.makedirs(self.save_dir, exist_ok=True)

        self.logs = self.save_dir + 'logs/'
        os.makedirs(self.logs, exist_ok=True)
        self.writer = SummaryWriter(self.logs)

        # Save train parameters
        with open('{}/train_params.txt'.format(self.save_dir), 'w') as f:
            json.dump(train_opt.__dict__, f, indent=2)

        # Read dataset
        dataset_configs = YamlRead(f'configs/dataset/{self.dataset}.yaml')
        self.train_dir = dataset_configs.train_dir
        self.val_dir = dataset_configs.val_dir
        self.mean = dataset_configs.mean
        self.std = dataset_configs.std
        self.pos_dic = dataset_configs.pos_dic
        self.max_sequence = dataset_configs.max_sequence
        self.num_cls = dataset_configs.num_cls

        #Read Model configs
        model_configs = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
        self.signal_size = model_configs.signal_size

        if isinstance(self.val_dir, float):
            list_dataset = []
            for file in os.listdir(self.train_dir):
                if file.endswith(".npy"):
                    label = os.path.join(self.train_dir, file[:-4] + '.txt')
                    if os.path.isfile(label):
                        list_dataset.append(file)
            train_list, val_list = train_test_split(list_dataset, test_size=self.val_dir, random_state=42)
        else:
            train_list = []
            val_list = []
            for file in os.listdir(self.train_dir):
                if file.endswith(".npy"):
                    label = os.path.join(self.train_dir, file[:-4] + '.txt')
                    if os.path.isfile(label):
                        train_list.append(file)

            for file in os.listdir(self.val_dir):
                if file.endswith(".npy"):
                    label = os.path.join(self.val_dir, file[:-4] + '.txt')
                    if os.path.isfile(label):
                        val_list.append(file)

        # Data Loader
        #Read Aug config
        augcfs = YamlRead(f'configs/aug/{train_opt.aug}.yaml')

        self.device = train_opt.device
        self.batch_size = train_opt.batch_size

        train_transforms = [
            tr.AddNoise(p=augcfs.p_noise, target_snr_db=augcfs.noise_db, mean_noise=0),
            tr.Revert(p=augcfs.p_revert),
            tr.MaskZeros(p=augcfs.p_mask, mask_p=augcfs.mask_size),
            tr.ShiftLR(p=augcfs.p_shiftLR, shift_p=augcfs.shiftLR_size),
            tr.ShiftUD(p=augcfs.p_shiftUD, shift_p=augcfs.shiftUD_size),
            tr.Normalizer(with_std=False),
            tr.Resizer(signal_size=self.signal_size)
        ]

        training_set = FCGClassificationDataset(root_dir=self.train_dir, list_data=train_list,
                                                max_sequence=self.max_sequence,
                                                transform=transforms.Compose(train_transforms))

        train_params = {
            'batch_size': self.batch_size,
            'shuffle': True,
            'drop_last': True,
            'num_workers': train_opt.num_worker
        }

        self.training_generator = DataLoader(training_set, collate_fn=tr.collater, **train_params)

        validation_transforms = [
            tr.Normalizer(with_std=False),
            tr.Resizer(signal_size=self.signal_size)
        ]

        val_set = FCGClassificationDataset(root_dir=self.val_dir, list_data=val_list,
                                           max_sequence=self.max_sequence,
                                           transform=transforms.Compose(validation_transforms))

        val_params = {
            'batch_size': self.batch_size,
            'shuffle': False,
            'drop_last': False,
            'num_workers': train_opt.num_worker
        }
        self.val_generator = DataLoader(val_set, collate_fn=tr.collater, **val_params)

        model = AutoModelForImageClassification.from_pretrained(self.model_name, trust_remote_code=True)

        # if torch.cuda.is_available():
        #     model = nn.DataParallel(model)
        if train_opt.ckpt is not None:
            weight = torch.load(train_opt.ckpt, map_location=self.device)
            model.load_state_dict(weight, strict=True)

        self.model = model
        self.model.to(self.device)

        # Optimizer and Learning rate scheduler

        self.opti = train_opt.optimizer
        self.l_rate = train_opt.lr
        self.lr_scheduler = train_opt.lr_scheduler

        if self.opti == 'adamw':
            self.optimizer = torch.optim.Adam(params=model.parameters(),
                                              lr=self.l_rate)
        else:
            self.optimizer = torch.optim.SGD(params=self.model.parameters(),
                                             lr=self.l_rate,
                                             momentum=0.9)

        if self.lr_scheduler == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=self.optimizer,
                                                                                  T_0=40,
                                                                                  T_mult=2)
        else:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=3, verbose=True)

        self.num_iter_per_epoch = len(self.training_generator)
        self.step = 0
        self.best_train_loss = 1e5
        self.best_val_loss = 1e5
        self.current_val_loss = 0.0
        self.epochs = train_opt.epochs
        self.total_fcng = np.zeros((1, 8))

        self.train_data_dis = np.zeros((1, self.num_cls))
        self.val_data_dis = np.zeros((1, self.num_cls))
        self.cls_weight = np.zeros((1, self.num_cls))
        self.loss_weight = np.zeros((1, self.num_cls))
        self.EarlyStop = EarlyStopper(patience=10, min_delta=0.1)

    def data_analysis(self):
        # Train data
        print("Analyze Training dataset")
        progress_bar = tqdm(self.training_generator)
        for iter, data in enumerate(progress_bar):
            signals, label = data['signal'], data['label']
            label = label.cpu().numpy()
            self.train_data_dis += np.sum(label, axis=0)
        plot_data(data_dis=self.train_data_dis, save_dir=self.save_dir, save_name="train_distribution.png")

        print("Analyze Validation dataset")
        progress_bar = tqdm(self.val_generator)
        for iter, data in enumerate(progress_bar):
            signals, label = data['signal'], data['label']
            label = label.cpu().numpy()
            self.val_data_dis += np.sum(label, axis=0)
        plot_data(data_dis=self.val_data_dis, save_dir=self.save_dir, save_name="val_distribution.png")

        sum = np.sum(self.train_data_dis, axis=1)
        self.cls_weight = self.train_data_dis / sum
        max = np.max(self.cls_weight)
        self.loss_weight = max / self.cls_weight
        # self.loss_weight = (1.0 / self.cls_weight)
        self.model = self.model.to(self.device)

    def train(self, epoch):
        self.model.train()
        last_epoch = self.step // self.num_iter_per_epoch
        progress_bar = tqdm(self.training_generator)
        epoch_loss = []

        for iter, data in enumerate(progress_bar):
            if iter < self.step - last_epoch * self.num_iter_per_epoch:
                progress_bar.update()
                continue
            try:
                signals, label = data['signal'], data['label']
                signals = signals.to(self.device)
                label = label.to(self.device)
                label = label.to(torch.float32)
                self.optimizer.zero_grad()

                loss = self.model(signals, label)['loss']

                epoch_loss.append(float(loss))

                loss.backward()

                self.optimizer.step()

                current_lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('learning_rate', current_lr, self.step)

                self.writer.add_scalars('Step_Loss', {'loss': loss}, self.step)

                epoch_loss.append(loss.item())

                descriptor = '[Train] Step: {}. Epoch: {}/{}. Iteration: {}/{}. Loss: {:.6f}.'.format(
                        self.step, epoch+1, self.epochs, iter + 1, self.num_iter_per_epoch, loss.item())
                progress_bar.set_description(descriptor)
                self.step += 1

                if self.lr_scheduler == 'cosine':
                    self.scheduler.step(epoch + iter / self.num_iter_per_epoch)

            except Exception as e:
                print('[Error]', traceback.format_exc())
                print(e)
                continue

        if self.lr_scheduler == 'reduce':
            self.scheduler.step(np.mean(epoch_loss))
        # if self.lr_scheduler == 'cosine':
        #     self.scheduler.step()

        mean_loss = np.mean(epoch_loss)

        if self.best_train_loss > mean_loss:
            self.best_train_loss = mean_loss
            self.save_checkpoint(self.model, self.save_dir, 'best_train_loss.pt')

        train_descrip = '[Train] Epoch: {}. Mean Loss: {:.6f}.'.format(epoch+1, mean_loss)
        print(train_descrip)
        self.writer.add_scalars('Loss', {'train': mean_loss}, epoch)

    def validation(self, epoch):
        self.model.eval()
        progress_bar = tqdm(self.val_generator)
        epoch_loss = []

        for iter, data in enumerate(progress_bar):
            with torch.no_grad():
                try:
                    signals, label = data['signal'], data['label']
                    signals = signals.to(self.device)
                    label = label.to(self.device)
                    label = label.to(torch.float32)

                    loss = self.model(signals, label)['loss']

                    epoch_loss.append(loss.item())

                    descriptor = '[Valid] Step: {}. Epoch: {}/{}. Iteration: {}/{}. Loss: {:.6f}.'.format(
                        epoch * len(progress_bar) + iter, epoch, self.epochs, iter + 1, len(progress_bar), loss.item())
                    progress_bar.set_description(descriptor)

                except Exception as e:
                    print('[Error]', traceback.format_exc())
                    print(e)
                    continue

        mean_loss = np.mean(epoch_loss)
        self.current_val_loss = mean_loss
        val_descrip = '[Validation] Epoch: {}. Mean Loss: {:.6f}.'.format(epoch + 1, mean_loss)
        print(val_descrip)

        self.writer.add_scalars('Loss', {'valid': mean_loss}, epoch)

        if self.nni_writer:
            nni.report_intermediate_result(mean_loss)
            if epoch == self.epochs - 1:
                nni.report_final_result(mean_loss)

        self.save_checkpoint(self.model, self.save_dir, 'last.pt')

        if self.best_val_loss > mean_loss:
            self.best_val_loss = mean_loss
            self.save_checkpoint(self.model, self.save_dir, 'best_val_loss.pt')

    def start(self):
        # self.data_analysis()
        for epoch in range(self.epochs):
            self.train(epoch)
            self.validation(epoch)
            early_stop = self.EarlyStop(self.current_val_loss)
            if early_stop:
                nni.report_final_result(self.current_val_loss)
                print("Early Stop at {} epochs".format(epoch))
                break

    def save_checkpoint(self, model, saved_path, name):
        torch.save(model.state_dict(), saved_path + name)
