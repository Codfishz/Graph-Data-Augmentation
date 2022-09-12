import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GINConv
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling

class ApplyNodeFunc(nn.Module):
    def __init__(self, mlp):
        super(ApplyNodeFunc,self).__init__()
        self.mlp=mlp
        self.bn=nn.BatchNorm1d(self.mlp.output_dim)
    def forward(self,h):
        h=self.mlp(h)
        h=self.bn(h)
        h=F.relu(h)
        return h
        
class MLP(nn.Module):
    def __init__(self,num_layers,input_dim,hidden_dim,output_dim):
        super(MLP,self).__init__()
        self.linear_or_not = True
        self.num_layers = num_layers
        self.output_dim = output_dim
        
        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))
            
            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))
        
    def forward(self,x):
        if self.linear_or_not:
            return self.linear(x)
        else:
            h=x
            for i in range(self.num_layers - 1):#除最后一层外都加一个relu
                h = F.relu(self.batch_norms[i](self.linears[i](h)))
            return self.linears[-1](h)
    
class GIN(nn.Module):
    def __init__(self,num_layers=5, num_mlp_layers=2, input_dim=150, hidden_dim=300,output_dim=150, dropout=0.5, learn_eps=False, graph_pooling_type="sum",neighbor_pooling_type="sum"):
                 
        super(GIN, self).__init__()
        self.num_layers = num_layers
        self.learn_eps=learn_eps
        
        self.ginlayers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        
        for layer in range(self.num_layers - 1):
            if layer==0:
                mlp = MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim)
            else:
                mlp = MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim)
            self.ginlayers.append(
                GINConv(ApplyNodeFunc(mlp), neighbor_pooling_type, 0, self.learn_eps))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        self.linears_prediction = torch.nn.ModuleList()
        for layer in range(self.num_layers):
            if layer==0:
                self.linears_prediction.append(
                    nn.Linear(input_dim, output_dim))
            else:
                self.linears_prediction.append(
                    nn.Linear(hidden_dim, output_dim))
        self.drop=nn.Dropout(dropout)
        
        if graph_pooling_type == 'sum':
            self.pool = SumPooling()
        elif graph_pooling_type == 'mean':
            self.pool = AvgPooling()
        elif graph_pooling_type == 'max':
            self.pool = MaxPooling()
        else:
            raise NotImplementedError
    def forward(self,g,h):
        hidden_rep = [h]
        for i in range(self.num_layers - 1):
            h = self.ginlayers[i](g, h)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            hidden_rep.append(h)
        
        score_over_layer = 0
        for i, h in enumerate(hidden_rep):
            pooled_h = self.pool(g, h)
            score_over_layer += self.drop(self.linears_prediction[i](pooled_h))
        return score_over_layer
        
        
