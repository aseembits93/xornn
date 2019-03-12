from common_utils import *
import ipdb as pdb
class SimpleNetwork(nn.Module):
    '''Network Definiton
    '''
    def __init__(self,config):
        '''
        Initialize Network with configuration
        '''
        super(SimpleNetwork,self).__init__()
        self.config = config
        self.fc1 = nn.Linear(self.config['ip_dim'],self.config['hl_dim'])
        self.fc2 = nn.Linear(self.config['hl_dim'],self.config['op_dim'])
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(self.config['hl_dim'])

    def forward(self,X):
        '''
        Forward Pass of Network
        '''
        X = self.fc1(X)
        X = self.relu(X)
        X = self.bn(X)
        Y = self.fc2(X)
        return Y

# class G(nn.module):
#     '''Generator network
#     '''
#     def __init__(self,config):
#         super(G,self).__init__()

# class D(nn.module):
#     '''Discriminator network
#     '''
#     def __init__(self,config):
#         super(D,self).__init__()


# class GAN(nn.Module):
#     '''Generative Adverserial Network
#     '''
#     def __init__(self,config):
#         super(GAN,self).__init__()
#         self.G = G()
#         self.D = D()

#     def forward(self,X):
#         X = self.G(X)

# class LSTMNetwork(nn.Module):
#     '''Network Definiton
#     '''
#     def __init__(self,config):
#         '''
#         Initialize Network with configuration
#         '''
#         super(LSTMNetwork,self).__init__()
#         self.lstm = nn.LSTM(1,10,2,bidirectional=True, dropout=0.5, batch_first=True)
#         self.fc1 = nn.Linear(20,20)   
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout()
#         self.fc2 = nn.Linear(20,2)

#     def forward(self,X):#,hd):
#         '''
#         Forward Pass of Network
#         '''
#         #pdb.set_trace()
#         Y, _ = self.lstm(X)#,hd)
#         #pdb.set_trace()
#         Y = self.fc1(Y[:,-1,:])
#         Y = self.relu(Y)
#         Y = self.dropout(Y)
#         Y = self.fc2(Y)
#         return Y

# class LSTMNetwork(nn.Module):
#     '''Network Definiton
#     '''
#     def __init__(self,config):
#         '''
#         Initialize Network with configuration
#         '''
#         super(LSTMNetwork,self).__init__()
#         self.lstm = nn.LSTM(1,1,batch_first=True)
#         self.conv1 = nn.Conv1d(1,1,2)
#         self.fc1 = nn.Linear(50,50)
#         self.bn = nn.BatchNorm1d(49)   
#         self.relu = nn.LogSigmoid()
#         #self.dropout = nn.Dropout()
#         self.fc2 = nn.Linear(49,2)

#     def forward(self,X):#,hd):
#         '''
#         Forward Pass of Network
#         '''
#         #pdb.set_trace()
#         Y, _ = self.lstm(X)#,hd)
        
#         Y = self.conv1(Y.view(100000,1,50))
#         Y = self.relu(Y)
#         #pdb.set_trace()
#         Y = self.bn(Y.view(100000,49))
#         Y = self.fc2(Y.view(100000,49))
#         #Y = self.relu(Y)
#         #Y = self.dropout(Y)
#         #Y = self.fc2(Y)
#         #pdb.set_trace()
#         return Y

class LSTMNetwork(nn.Module):
    '''Network Definiton
    '''
    def __init__(self,config):
        '''
        Initialize Network with configuration
        '''
        super(LSTMNetwork,self).__init__()
        self.lstm = nn.LSTM(1,1,batch_first=True)
        for name,value in self.lstm.named_parameters():
            if 'weight' in name:
                #pdb.set_trace()
                nn.init.xavier_normal_(value)
            else:
                nn.init.constant_(value,1)
        #self.conv1 = nn.Conv1d(1,1,2)
        self.fc1 = nn.Linear(17,17)
        #self.bn = nn.BatchNorm1d(49)   
        self.relu = nn.ReLU()
        #self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(17,2)

    def forward(self,X):#,hd):
        '''
        Forward Pass of Network
        '''
        Y, _ = self.lstm(X)#,hd)
        
        Y = Y.squeeze()
        Y = self.fc1(Y)
        Y = self.relu(Y)
        Y = self.fc2(Y)
        return Y        

class VariableLSTMNetwork(nn.Module):
    '''Network Definiton
    '''
    def __init__(self,config):
        '''
        Initialize Network with configuration
        '''
        super(LSTMNetwork,self).__init__()
        self.lstm = nn.LSTM(1,2,batch_first=True)
        for name,value in self.lstm.named_parameters():
            if 'weight' in name:
                #pdb.set_trace()
                nn.init.xavier_normal_(value)
            else:
                nn.init.constant_(value,1)
        #self.conv1 = nn.Conv1d(1,1,2)
        #self.fc1 = nn.Linear(17,17)
        #self.bn = nn.BatchNorm1d(49)   
        #self.relu = nn.ReLU()
        #self.dropout = nn.Dropout()
        #self.fc2 = nn.Linear(17,2)

    def forward(self,X):#,hd):
        '''
        Forward Pass of Network
        '''
        Y, _ = self.lstm(X)#,hd)
        pdb.set_trace()
        Y = Y.squeeze()
        #Y = self.fc1(Y)
        #Y = self.relu(Y)
        #Y = self.fc2(Y)
        return Y[:,-1,:]        