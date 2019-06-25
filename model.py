# Implement baseline models here.

from math import floor
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

def count_parameters(model):
    total_param = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_param = np.prod(param.size())
            # if param.dim() > 1:
            #     print(name, ':', 'x'.join(str(x) for x in list(param.size())), '=', num_param)
            # else:
            #     print(name, ':', num_param)
            total_param += num_param
    # print('Total Parameters: {}'.format(total_param))
    return total_param
    
class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()
        self.FC = nn.Sequential(
            nn.Linear(1280, 512),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            # nn.Linear(512, 512),
            # nn.Dropout(0.5),
            # nn.LeakyReLU(),
            # nn.Linear(512, 512),
            # nn.Dropout(0.5),
            # nn.LeakyReLU(),
            nn.Linear(512, 128),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.Linear(128, 20)
        )
        self.param_count = count_parameters(self)
        print(self.param_count)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
    def forward(self, X):
        X = X.view(X.shape[0], -1)
        out = torch.sigmoid(self.FC(X))
        return out
     
# I used this     
class FC_T(nn.Module):
    def __init__(self):
        super(FC_T, self).__init__()
        self.embed = nn.Sequential(
            nn.Linear(128, 128),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            # nn.Linear(512, 512),
            # nn.Dropout(0.5),
            # nn.LeakyReLU(),
            # nn.Linear(512, 512),
            # nn.Dropout(0.5),
            # nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
        )
        
        self.classify = nn.Linear(128, 20)
        self.param_count = count_parameters(self)
        print(self.param_count)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
    def forward(self, X):
        out = self.embed(X)
        out = (out+X).mean(1)
        out = torch.sigmoid(self.classify(out))
        return out
        
# Same model as above with max pooling instead of mean pooling
# class FC_T(nn.Module):
    # def __init__(self):
        # super(FC_T, self).__init__()
        # self.embed = nn.Sequential(
            # nn.Linear(128, 128),
            # nn.Dropout(0.6),
            # nn.LeakyReLU(),
            # nn.Linear(128, 128),
            # nn.Dropout(0.6),
            # nn.LeakyReLU(),
            # # nn.Linear(512, 512),
            # # nn.Dropout(0.5),
            # # nn.LeakyReLU(),
            # # nn.Linear(512, 512),
            # # nn.Dropout(0.5),
            # # nn.LeakyReLU(),
            # nn.Linear(128, 128),
            # nn.Dropout(0.6),
            # nn.LeakyReLU(),
        # )
        
        # self.classify = nn.Linear(128, 20)
        # self.param_count = count_parameters(self)
        # print(self.param_count)
        
        # for m in self.modules():
            # if isinstance(m, nn.Linear):
                # init.xavier_normal_(m.weight.data)
                # if m.bias is not None:
                    # m.bias.data.zero_()
    # def forward(self, X):
        # out = self.embed(X)
        # out,_ = (out+X).max(1)
        # out = torch.sigmoid(self.classify(out))
        # return out


class BaselineRNN_2(nn.Module):
    def __init__(self):
        super(BaselineRNN_2, self).__init__()
        self.rnn = nn.GRU(128, 64, num_layers=3, bidirectional=True, dropout=0.5, batch_first=True)
        self.FC = nn.Linear(128, 20)
        self.param_count = count_parameters(self)
        print(self.param_count)
    def forward(self, X):
        out, _ = self.rnn(X)
        out = torch.sigmoid(self.FC(out[:,-1,:]))
        return out
