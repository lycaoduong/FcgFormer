"""
This is pytorch reimplemented IRCNN model based on "Automatic materials characterization from infrared
spectra using convolutional neural networks" paper

Original Model (Keras Based): https://github.com/gj475/irchracterizationcnn

Author: lycaoduong - 2024.01.04
"""
import torch.nn as nn
import torch


class IrCNN(nn.Module):
    def __init__(self, signal_size=1024, kernel_size=11, in_ch=1, num_cls=17, p=0.48599073736368):
        super(IrCNN, self).__init__()
        self.num_cls = num_cls
        # 1st CNN layer.
        self.CNN1 = nn.Sequential(
            nn.Conv1d(in_channels=in_ch, out_channels=31, kernel_size=kernel_size, stride=1, padding=0),
            nn.BatchNorm1d(num_features=31),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.cnn1_size = int(((signal_size - kernel_size + 1 - 2) / 2) + 1)
        # 2nd CNN layer.
        self.CNN2 = nn.Sequential(
            nn.Conv1d(in_channels=31, out_channels=62, kernel_size=kernel_size, stride=1, padding=0),
            nn.BatchNorm1d(num_features=62),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.cnn2_size = int(((self.cnn1_size - kernel_size + 1 - 2) / 2) + 1)
        # 1st dense layer.
        self.DENSE1 = nn.Sequential(
            nn.Linear(in_features=self.cnn2_size*62, out_features=4927),
            nn.ReLU(),
            nn.Dropout(p=p)
        )
        # 2st dense layer.
        self.DENSE2 = nn.Sequential(
            nn.Linear(in_features=4927, out_features=2785),
            nn.ReLU(),
            nn.Dropout(p=p)
        )
        # 3st dense layer.
        self.DENSE3 = nn.Sequential(
            nn.Linear(in_features=2785, out_features=1574),
            nn.ReLU(),
            nn.Dropout(p=p)
        )
        # FCN layer
        self.FCN = nn.Linear(in_features=1574, out_features=num_cls)

    def forward(self, signal):
        x = self.CNN1(signal)
        x = self.CNN2(x)
        x = torch.flatten(x, -2, -1)
        x = torch.unsqueeze(x, dim=1)
        x = self.DENSE1(x)
        x = self.DENSE2(x)
        x = self.DENSE3(x)
        x = self.FCN(x)
        # Comment it if Loss Function has activation function
        # x = torch.sigmoid(x)
        return x


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    model = IrCNN(signal_size=600)
    model = model.to(device)
    rs = torch.randn(8, 1, 600).to(device)
    o = model(rs)
    print(o)
