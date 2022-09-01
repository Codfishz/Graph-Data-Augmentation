import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from torch_geometric.data import Data
import dgl
import os
from torch.utils.data import Dataset
from sklearn import metrics
#import faulthandler
#faulthandler.enable()


class MRIDataset(Dataset):
    def __init__(self):
        
        X_set=pd.read_csv('/Users/Rui/Documents/GitHub/Graph-Data-Augmentation/Classification/adni_data_all.csv',engine="python")
        Y_label = X_set.loc[:,['DX.bl']]
        ImageID_all =X_set.loc[:,['ImageID']]
        train_data_cthick=np.zeros((len(Y_label),1284))
        Y_label_num=np.zeros((len(Y_label),1))
        #print(Y_label.iat[17,0])
        for i in range(len(Y_label)):
            if Y_label.iat[i,0]=="CN":
                Y_label_num[i,0]=0
            elif Y_label.iat[i,0]=="LMCI" or Y_label.iat[i,0]=="EMCI":
                Y_label_num[i,0]=1
            elif Y_label.iat[i,0]=="AD":
                Y_label_num[i,0]=2
        #    train_data_cthick[i,0:1284]=pd.read_table("/Users/Rui/Documents/MATLAB/thickness1k/mnc_118673_native_rms_rsl_tlink_20mm.txt",header=None, index_col=False).values
        train_data_cthick = np.array([pd.read_table("/Users/Rui/Documents/MATLAB/thickness1k/mnc_"+ str(id)+"_native_rms_rsl_tlink_20mm.txt",header=None, index_col=False).values for id in X_set['ImageID']]).reshape(len(Y_label),1284)
        #train_data_cthick = np.array([pd.read_table("/Users/Rui/Documents/MATLAB/thickness1k/mnc_118673_native_rms_rsl_tlink_20mm.txt",header=None, index_col=False).values for id in X_set['ImageID']]).reshape(len(Y_label),1284)
        #print(train_data_cthick.shape)#(6424,1284)


        edges1k=pd.read_table('/Users/Rui/Documents/MATLAB/edges1k.txt',names=['node1','node2'],encoding='UTF-8')
        src, dst = tuple(zip(*np.array(edges1k)-1))

        print(len(src))
        print(len(dst))
        g=dgl.graph((src,dst))
        #g = dgl.DGLGraph()
        #g.add_nodes(1284)
        #g.add_edges(src, dst)
        g.add_edges(dst, src)
        nx_G = g.to_networkx().to_undirected()
        #g.ndata['DX.bl']=train_data_cthick # require(1284,1)?
        #print(g.ndata['DX.bl'])
        print('We have %d nodes.' % g.number_of_nodes())
        print('We have %d edges.' % g.number_of_edges())
        print('We have %d nodes.' % nx_G.number_of_nodes())
        print('We have %d edges.' % nx_G.number_of_edges())
        
        self.labels=Y_label_num
        self.edges=edges1k
        self.cthink=train_data_cthick
        self.sampleGraph=nx_G
        
        
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.cthink[idx], self.labels[idx]

#if __name__ == '__main__':
        
        
class Evaluator():
    def eval(input_dict):
        label=input_dict["y_true"]
        pred=input_dict["y_pred"]
        rocauc = metrics.roc_auc_score(label, score)
        return rocauc
        
