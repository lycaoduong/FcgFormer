import os.path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd


funcs_name = ["alkane", "methyl", "alkene", "alkyne", "alcohols", "amines", "nitriles", "aromatics",
              "alkyl halides", "esters", "ketones", "aldehydes", "carboxylic acids",
              "ether", "acyl halides", "amides", "nitro"]

num_data = ["1-Group", "2-Group", "3-Group", "4-Group", "5-Group", "6-Group", "7-Group"]


def plot_cross_attention_map(spectra, result, att_map):
    if att_map is not None:
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax2 = plt.twinx().twiny()
        ax2.set_xlim(0, len(spectra))
        ax1.set_yticks(np.arange(len(funcs_name)), labels=funcs_name)
        # ax1.set_xlim(0, 64)

        tem = np.zeros_like(att_map)
        for i, r in enumerate(funcs_name):
            if r in result:
                tem[i, :] = att_map[i, :]

        ax1.imshow(tem[:, :], cmap='viridis', interpolation='nearest', aspect="auto")
        ax2.plot(spectra)
        plt.show()


def plot_self_attention_map(spectra, att_map, offset=400):
    if att_map is not None:
        fig, ax1 = plt.subplots(figsize=(12, 12))
        ax2 = plt.twinx().twiny()
        ax2.set_xlim(offset, offset+len(spectra))
        ax1.set_xlabel('Patch index', fontsize=18)
        ax1.set_ylabel('Patch index', fontsize=18)
        ax1.imshow(att_map[1:, 1:], cmap='inferno', interpolation='nearest', aspect="auto")
        # tem_x = np.zeros_like(spectra)
        x = np.linspace(offset, offset + len(spectra), len(spectra))
        # for i, s in enumerate(spectra):
        #     tem_x[i] = i + 400
        ax2.set_xlabel('Wavelength', fontsize=18)
        ax2.set_ylabel('Intensity (a.u.)', fontsize=18)
        ax2.plot(x, spectra)
        plt.show()

def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i, y[i], y[i], ha='center')

def plot_data(data_dis, save_dir="./", save_name="fg.png"):
    plt.rc('font', size=20)
    fig = plt.figure(figsize=(20, 10))
    x_pos = np.arange(len(funcs_name))
    plt.bar(funcs_name, data_dis[0], align='center', alpha=0.5)
    addlabels(x_pos, data_dis[0])
    plt.xticks(x_pos, funcs_name, rotation=45)
    plt.ylabel('Total samples', fontsize=30)
    plt.title('Functional Groups Data Distribution', fontsize=25)
    # plt.show()
    save_path = os.path.join(save_dir, save_name)
    plt.tight_layout()
    plt.savefig(save_path)


def plot_numF_Data(data_dis, save_dir="./", save_name="fg.png"):
    plt.rc('font', size=20)
    fig = plt.figure(figsize=(20, 10))
    x_pos = np.arange(len(num_data))
    plt.bar(num_data, data_dis[0], align='center', alpha=0.5)
    addlabels(x_pos, data_dis[0])
    plt.xticks(x_pos, num_data, rotation=45)
    plt.ylabel('Total samples', fontsize=30)
    plt.title('Number of Functional Groups Data Distribution', fontsize=25)
    # plt.show()
    save_path = os.path.join(save_dir, save_name)
    plt.tight_layout()
    plt.savefig(save_path)

def plot_roc_pr_curve(precision, recall, fpr, save_dir="./", save_name="funcs_roc_pr_curve.png"):
    fig = plt.figure(figsize=(24, 12))
    plt.style.use('ggplot')
    # Add value
    precision.insert(0, 0.0001)
    recall.insert(0, 0.9999)
    fpr.insert(0, 0.9999)

    precision.append(0.9999)
    recall.append(0.0001)
    fpr.append(0.0001)

    # ROC curve
    plt.subplot(121)
    plt.plot(fpr, recall, linewidth=2)
    plt.title('ROC Curve', fontsize=40, fontweight="bold", y=1.05)
    plt.fill_between(fpr, recall, facecolor='blue', alpha=0.1)
    plt.text(0.55, 0.4, 'AUC', fontsize=40)
    # styling figure
    plt.xlabel('False Positive Rate', fontsize=35, labelpad=13)
    plt.ylabel('True Positive Rate', fontsize=35, labelpad=13)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim(-0.01, 1.01)
    plt.ylim(-0.01, 1.01)

    # PR Curve
    plt.subplot(122)
    plt.plot(recall, precision)
    plt.title('PR Curve', fontsize=40, fontweight="bold", y=1.05)
    plt.ylabel('Precision', fontsize=35, labelpad=13)
    plt.xlabel('Recall', fontsize=35, labelpad=13)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.xlim(-0.01, 1.01)
    plt.ylim(-0.01, 1.01)
    # plt.show()
    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path)
    plt.tight_layout()
    plt.close()

def plot_conf(conf, labelX=["1", "0"], labelY=["1", "0"], title=None, save_dir="./", save_name='fg.png', size=None, rotationY=0):
    # font = {'size': 11}
    plt.rc('font', size=30)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    # sns.set(font_scale=1.8)
    if size is None:
        fig = plt.figure(figsize=(15, 12))
    else:
        fig = plt.figure(figsize=size)
    plt.style.use('classic')
    # disp = ConfusionMatrixDisplay(confusion_matrix=conf, display_labels=label)
    # disp.plot(values_format='')
    ax = plt.subplot()
    sns.heatmap(conf, annot=True, fmt='g', ax=ax, cmap='Blues', annot_kws={'size': 30})
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)# annot=True to annotate cells, ftm='g' to disable scientific notation
    # labels, title and ticks
    ax.set_xlabel('Predicted labels', fontsize=35)
    ax.set_ylabel('True labels', fontsize=35)
    # ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(labelX, fontsize=30)
    ax.yaxis.set_ticklabels(labelY, fontsize=30, rotation=rotationY)

    if title is None:
        ax.set_title('Confusion Matrix', fontsize=40)
    else:
        ax.set_title(title, fontsize=40)
        # plt.title(title)
    # plt.show()
    save_path = os.path.join(save_dir, save_name)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def func_confusion(target, result, th=0.5):
    num_cls = len(funcs_name)
    conf_matrix = np.zeros((num_cls, 2, 2))
    target = target.astype(np.uint8)
    result = (result >= th).astype(np.uint8)
    for idx, name in enumerate(funcs_name):
        cf = confusion_matrix(target[:, idx], result[:, idx], labels=[1, 0])
        conf_matrix[idx, :] = cf
    return conf_matrix

def subs_confusion(target, result, th=0.5):
    conf_matrix = np.zeros((1, 2))
    target = target.astype(np.uint8)
    result = (result >= th).astype(np.uint8)
    batch_size = target.shape[0]
    for idx in range(batch_size):
        tg = target[idx]
        pd = result[idx]
        res = np.array_equal(tg, pd)
        if res:
            conf_matrix[0, 0] += 1
        else:
            conf_matrix[0, 1] += 1
    return conf_matrix

def subs_len_confusion(target, result, th=0.5, fn=None):
    conf_matrix = np.zeros((8, 2))
    target = target.astype(np.uint8)
    result = (result >= th).astype(np.uint8)
    batch_size = target.shape[0]
    list_true = []
    for idx in range(batch_size):
        tg = target[idx]
        pd = result[idx]
        res = np.array_equal(tg, pd)
        len = np.sum(tg)
        if res:
            conf_matrix[len-1, 0] += 1
            conf_matrix[7, 0] += 1
            if fn is not None:
                list_true.append(fn[idx])
        else:
            conf_matrix[len-1, 1] += 1
            conf_matrix[7, 1] += 1
    return conf_matrix, list_true

def plot_loss_from_csv(fcg_loss, ircnn_loss, lr_dir):
    font = {'size': 11}
    plt.rc('font', **font)

    fcg_loss = pd.read_csv(fcg_loss)
    fcg_loss_value = fcg_loss['Value'].tolist()
    ircnn_loss = pd.read_csv(ircnn_loss)
    ircnn_loss_value = ircnn_loss['Value'].tolist()

    lr = pd.read_csv(lr_dir)
    lr_value = lr['Value'].tolist()

    fig, ax = plt.subplots(2, 1, figsize=(12, 10))

    ax[0].plot(fcg_loss_value, '-b', label='Fcg-Former')
    ax[0].plot(ircnn_loss_value, '-r', label='IRCNN')
    # ax[0].axis('equal')

    y_fgc, x_fcg = min(fcg_loss_value), fcg_loss_value.index(min(fcg_loss_value))
    ax[0].annotate('best checkpoint', xy=(x_fcg, y_fgc), xytext=(x_fcg-60, y_fgc+0.5),
                arrowprops={'arrowstyle': '->', 'ls': 'dashed', 'color': 'red'}, va='center')

    y_ircnn, x_ircnn = min(ircnn_loss_value), ircnn_loss_value.index(min(ircnn_loss_value))
    ax[0].annotate('best checkpoint', xy=(x_ircnn, y_ircnn), xytext=(x_ircnn-50, y_ircnn+0.5),
                arrowprops={'arrowstyle': '->', 'ls': 'dashed', 'color': 'red'}, va='center')

    y_ircnn_stop, x_ircnn_stop = ircnn_loss_value[-1], len(ircnn_loss)
    ax[0].annotate('early stop point', xy=(x_ircnn_stop, y_ircnn_stop), xytext=(x_ircnn_stop+50, y_ircnn_stop+0.5),
                arrowprops={'arrowstyle': '->', 'ls': 'dashed', 'color': 'black'}, va='center')


    # ax[0].annotate('learning rate restart point', xy=(120, fcg_loss_value[120+1]), xytext=(x_ircnn_stop+150, ircnn_loss_value[120]+0.5),
    #             arrowprops={'arrowstyle': '->', 'ls': 'dashed', 'color': 'black'}, va='center')

    ax[0].annotate('learning rate restart point', xy=(280, fcg_loss_value[280+1]), xytext=(x_ircnn_stop+150, ircnn_loss_value[120]+0.5),
                arrowprops={'arrowstyle': '->', 'ls': 'dashed', 'color': 'black'}, va='center')

    ax[0].set_title('Validation Loss', fontsize=18)
    ax[0].set_xlabel('Epochs', fontsize=16)
    ax[0].set_ylabel('Loss', fontsize=16)
    ax[0].set_xlim([0, 600])
    ax[0].set_ylim([0, 1.2])

    x = np.linspace(0, 600, len(lr_value))
    ax[1].plot(x, lr_value, label='Learning rate')
    ax[1].set_title('Learning rate scheduler', fontsize=18)
    ax[1].set_xlabel('Epochs', fontsize=16)
    ax[1].set_ylabel('Learning rate', fontsize=16)
    ax[1].set_xlim([0, 600])
    ax[1].set_ylim([0, 0.00025])

    leg = ax[0].legend()
    leg2 = ax[1].legend()
    fig.tight_layout(pad=3.0)
    plt.show()


if __name__ == '__main__':
    # tg = np.array([[1, 1, 1, 1], [0, 1, 1, 1], [0, 1, 1, 0]])
    # rs = np.array([[0.9, 0.8, 0.75, 0.9], [0.1, 0.8, 0.75, 0.9], [0.9, 0.9, 0.9, 0.1]])
    # cf = subs_len_confusion(tg, rs, th=0.5)
    fcg_p = 'D:/lycaoduong/workspace/paper/fcg-former/loss/fcg/run-Loss_val-tag-Loss.csv'
    lr_p = 'D:/lycaoduong/workspace/paper/fcg-former/loss/fcg/run-.-tag-learning_rate.csv'
    ircnn_p = 'D:/lycaoduong/workspace/paper/fcg-former/loss/ircnn/run-Loss_val-tag-Loss.csv'
    plot_loss_from_csv(fcg_p, ircnn_p, lr_p)