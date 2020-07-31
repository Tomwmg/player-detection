import torch
import torch.utils.data.dataset as Dataset
import numpy as np
import json
torch.manual_seed(123)
np.random.seed(123)

torch.backends.cudnn.deterministic = True

if torch.cuda.is_available():
   torch.cuda.manual_seed_all(123)


class YC2_train_data(Dataset.Dataset):
    def __init__(self,pos_feature,neg_feature,train=True):
        self.pos_feature=np.load(pos_feature)
        self.neg_feature=np.load(neg_feature)

        self.pos_max = self.pos_feature.shape[0]
        self.neg_max = self.neg_feature.shape[0]

    def __len__(self):
        return self.neg_max

    def __getitem__(self, index):
        neg_feature = self.neg_feature[index]
        pos_index=np.random.randint(0,self.pos_max)
        pos_feature = self.pos_feature[pos_index]
        pos_feature=torch.FloatTensor(pos_feature)
        neg_feature=torch.FloatTensor(neg_feature)
        result = [pos_feature, neg_feature]
        return result



def get_dataset():
    train_set = YC2_train_data( pos_feature='data/pos.npy',
                                neg_feature='data/neg.npy')

    return train_set

if __name__ == "__main__":
    from torch.utils.data import Dataset, DataLoader
    train_set = get_dataset()
    train_loader = DataLoader(train_set, batch_size=2, shuffle=True, num_workers=1)
    for train_data in train_loader:
        pos_feature,  neg_feature, =train_data
        print(pos_feature.shape,  neg_feature.shape, )




