from torch.utils.data import Dataset,DataLoader
import torch

class Normalize(object):
    def __init__(self, type = "0-1"): # "0-1","1-1","mean-std"
        self.type = type
    def __call__(self, seq):
        if  self.type == "0-1":
            seq = (seq-seq.min())/(seq.max()-seq.min())
        elif  self.type == "1-1":
            seq = 2*(seq-seq.min())/(seq.max()-seq.min()) + -1
        elif self.type == "mean-std" :
            seq = (seq-seq.mean())/seq.std()
        elif self.type=='None':
            seq=seq
        elif self.type=='mean':
            seq = seq - seq.mean()
        else:
            raise NameError('This normalization is not included!')
        return seq

class SigDataset(Dataset):
    def __init__(self, signal,Normalization):
        super(SigDataset,self).__init__()
        self.signal=torch.tensor(signal,dtype=torch.float32)
        self.transforms = Normalize(Normalization)
    def __len__(self):
        return len(self.signal)
    def __getitem__(self, idx):
        self.signal = self.transforms(self.signal)
        return self.signal[idx]

def LoadSig(signal,batch_size=32,Normalization='None'):
    dataset=SigDataset(signal,Normalization)
    print('{} samples found'.format(len(dataset)))
    train_iterator=DataLoader(dataset,batch_size,shuffle=False)
    return train_iterator


class SigDataset_N(Dataset):
    def __init__(self, signal,signal_N,Normalization):
        super(SigDataset_N,self).__init__()
        self.signal=torch.tensor(signal,dtype=torch.float32)
        self.signal_N=torch.tensor(signal_N,dtype=torch.float32)
        self.transforms = Normalize(Normalization)
    def __len__(self):
        return len(self.signal)
    def __getitem__(self, idx):
        self.signal = self.transforms(self.signal)
        self.signal_N = self.transforms(self.signal_N)
        return self.signal[idx],self.signal_N[idx]

def LoadSig_N(signal,signal_N,batch_size=32,Normalization='None'):
    dataset=SigDataset_N(signal,signal_N,Normalization)
    print('{} samples found'.format(len(dataset)))
    train_iterator=DataLoader(dataset,batch_size,shuffle=False)
    return train_iterator