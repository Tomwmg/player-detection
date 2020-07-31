# general packages
import numpy as np
import random
import os
import errno
import argparse
import torch
from torch.utils import model_zoo
from torch.autograd import Variable
from torch.optim import lr_scheduler

from data_loader import YC2_train_data
from model import DVSA
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from config import Config

torch.manual_seed(123)
np.random.seed(123)
torch.backends.cudnn.deterministic = True
CUDA = True if torch.cuda.is_available() else False


def pause():
    programPause = input("Press the <ENTER> key to continue...")


if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)


def adjust_learning_rate(optimizer, epoch, drop_rate, step):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.learning_rate * (drop_rate ** (epoch // step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main(args):
    try:
        os.makedirs(args.checkpoint_path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            print('Directory already exists.')
        else:
            raise
    print('loading dataset')

    print('building model')
    model = get_model(args)
    model.cuda()

    checkpoint = torch.load('demo.pth')
    model.load_state_dict(checkpoint)
    model.eval()
    pos_num=0
    neg_num=0
    num=0

    pos_feature=np.load('pos.npy')
    neg_feature=np.load('neg.npy')
    pos_feature=torch.FloatTensor(pos_feature)
    neg_feature = torch.FloatTensor(neg_feature)
    pos_feature = Variable(pos_feature).cuda()
    neg_feature = Variable(neg_feature).cuda()
    pos = model.score(pos_feature).data.cpu().numpy()
    neg = model.score(neg_feature).data.cpu().numpy()

    print(pos.shape[1],neg.shape[1],np.sum(pos),np.sum(neg))

def get_model(args):
    model = DVSA()
    return model



if __name__ == "__main__":
    args = Config()
    main(args)
