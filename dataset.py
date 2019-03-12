from common_utils import *
import ipdb as pdb 

class XORDataset(Dataset):

    def __init__(self, config, transform=None):
        
        self.data_points = 100000
        self.X = torch.rand((self.data_points,config['ip_dim']))
        self.Y = torch.LongTensor([bool(int(round(float(X1)))) ^ bool(int(round(float(X2)))) for X1,X2 in self.X]) #looks real ugly
        

    def __len__(self):
        
        return self.data_points

    def __getitem__(self, idx):
        
        return self.X[idx], self.Y[idx]

class LSTMXORDataset(Dataset):

    def __init__(self, config, transform=None):
        
        self.data_points = 100000
        self.str_len = 17
        self.X = torch.randint(0,2,(self.data_points,self.str_len,1))
        self.Y = self.X.sum(1) % 2
        self.X = self.X.type('torch.FloatTensor')
        self.Y.squeeze_()
        #pdb.set_trace()

    def __len__(self):
        
        return self.data_points

    def __getitem__(self, idx):
        
        return self.X[idx], self.Y[idx]       

class VariableLSTMXORDataset(Dataset):

    def __init__(self, config, transform=None):
        
        self.data_points = 100000
        self.str_lens, _ = torch.sort(torch.randint(1,51,(data_points,)),descending=True)
        self.X = [torch.randint(0,2,(l,)) for l in self.str_lens]
        self.Y = torch.tensor([seq.sum() % 2 for seq in self.X],dtype=torch.float32)
        self.X = torch.nn.utils.rnn.pack_sequence(self.X)
        #pdb.set_trace()

    def __len__(self):
        
        return self.data_points

    def __getitem__(self, idx):
        
        return self.X[idx], self.Y[idx]               