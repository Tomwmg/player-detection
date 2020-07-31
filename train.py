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
    train_set, val_set = get_dataset(args)
    train_loader = DataLoader(train_set, batch_size = 10, shuffle = True, num_workers= 6)
    #test_loader  = DataLoader(val_set, batch_size=1, shuffle = False, num_workers = 1)

    print('building model')
    model = get_model(args)
    model.cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learn_rate, weight_decay=1e-4)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size= 30*len(train_loader), gamma = 1e-1)
    if args.start_from!='':
        print('start from args.start_from')
        checkpoint = torch.load(args.start_from)
        model.load_state_dict(checkpoint)
    'start training'
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        epoch_loss = 0.
        loss_time=0

        output_str = "%s " %(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        for batch_cnt, batch in enumerate(train_loader):

            pos_feature,  neg_feature, = batch
            pos_feature = Variable(pos_feature).cuda()
            neg_feature = Variable(neg_feature).cuda()
            cost = model(pos_feature, neg_feature)
            optimizer.zero_grad()
            exp_lr_scheduler.step()     
            cost.backward()
            optimizer.step()
            epoch_loss += cost.item()
            loss_time+=1



        print(output_str, ' epoch: %d,  epoch_loss: %lf, lr: %lf' % (epoch, epoch_loss / loss_time, optimizer.param_groups[0]['lr']))
        if (epoch+1)%5==0:
            torch.save(model.state_dict(),
                       args.checkpoint_path + '/' + '%d'%epoch + '.pth')

def get_model(args):
    model=DVSA()
    return model
def get_dataset(args):
    train_set = YC2_train_data(pos_feature='data/pos.npy',
                               neg_feature='data/neg.npy')
    val_set = YC2_train_data(pos_feature='data/pos.npy',
                            neg_feature='data/neg.npy',train=False)

    return train_set,val_set

if __name__ == "__main__":
    args = Config()
    main(args)
