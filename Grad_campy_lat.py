import networkx as nx
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import spektral
import os, time, collections, shutil, ntpath
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
import lib3
import lib2
#from lib2 import models_vae, coarsening, graph
import pandas as pd
import numpy as np
import dgl 
import tensorflow as tf
import torch as th
from spektral.data.loaders import SingleLoader
from spektral.datasets.citation import Citation
from spektral.layers import GCNConv
from spektral.models.gcn import GCN
from sklearn.model_selection import train_test_split
from spektral.transforms import AdjToSpTensor, LayerPreprocess
from tensorflow.keras.layers import Input, Dense, Flatten
from spektral.utils.sparse import sp_matrix_to_sp_tensor
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from spektral.datasets import Citation
from spektral.utils import one_hot, normalized_laplacian
from imblearn.over_sampling import SMOTE
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
import heapq
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve,auc
import LabTool
import h5py
from collections import Counter
import scipy.io

def count_intersections(lst1, lst2):
    c1 = Counter(lst1.tolist())
    c2 = Counter(lst2.tolist())
    intersect=np.zeros(np.intersect1d(lst1,lst2).shape,dtype=int)
    freq=np.zeros(np.intersect1d(lst1,lst2).shape,dtype=int)
    count=0
    for k in c1.keys() & c2.keys():
        freq[count]=min(c1[k], c2[k])
        intersect[count]=k
        count+=1
    return intersect,freq;


def getplot(cross,z,label):
    z=z-1
    c=[]
    for i in range(1,len(label)+1,1):
         if(np.array(label)[i-1,0]==1):
                c.append('r')
         else:
                c.append('g')
    aa=range(1,len(cross)+1,1)
    fig, ax=plt.subplots(figsize=(16,4 ))
    plt.bar(aa,cross[z,:],label=label,color=c,tick_label=c)
    for a in aa:
       plt.text(a, 51,'%.0f' % a, ha='center', va='bottom', fontsize=10)
    plt.xticks(aa,size='medium',rotation=30)
    print('this data point is',label[z][0])
    print('the color is',c[z])

def plotcurve(history,a):
  plt.figure(1)
  fig, ax = plt.subplots(1,2,figsize = (78,25),linewidth=30)
  ax[0].plot(history.history['loss'],color='#EFAEA4',label = 'Training Loss',linewidth=15)
  ax[0].plot(history.history['val_loss'],color='#B2D7D0',label = 'Test Loss',linewidth=15)
  ax[1].plot(history.history['acc'],color='#EFAEA4',label = 'Training Accuracy',linewidth=15)
  ax[1].plot(history.history['val_acc'],color='#B2D7D0',label = 'Test Accuracy',linewidth=15)
  ax[0].legend(prop={"size":66},loc='upper right')
  ax[1].legend(prop={"size":66},loc='lower right')
  ax[0].set_xlabel('Epochs',fontsize=76)
  ax[1].set_xlabel('Epochs',fontsize=76);
  ax[0].set_ylabel('Loss',fontsize=76)
  ax[1].set_ylabel('Accuracy %',fontsize=76);
  ax[0].title.set_size(95)
  ax[1].title.set_size(95)
  #ax.setxtickss([0,1])
  ax[0].set_yticklabels([0.1,0.2,0.4,0.6,0.8,1.0,3.0,4.0],fontsize=40)
  ax[0].set_xticklabels([0,20,40,60,80,100],fontsize=40)
  plt.xticks(fontsize=40)
  plt.yticks(fontsize=40)


def modelBuilder(l2_reg,learning_rate,gcn_num):     
    N=1284
    F=1
    n_out=1
    X_in = Input(shape=(N,F))
    A_in = Input(shape=(N,N))
    graph_conv = GCNConv(10,
    activation='relu',
    kernel_regularizer=l2(l2_reg),
    use_bias=True)([X_in, A_in])
    graph_conv2 = GCNConv(10,
    activation='relu',
    kernel_regularizer=l2(l2_reg),
    use_bias=True)([graph_conv, A_in])
    graph_conv3 = GCNConv(gcn_num,
    activation='relu',
    kernel_regularizer=l2(l2_reg),
    use_bias=True)([graph_conv2, A_in])
    fc = Flatten()(graph_conv3)
    dense1=Dense(7,activation='relu')(fc)
    dp0=Dropout(0.2)(dense1)
    dense3=Dense(5,activation='relu')(dp0)
    output = Dense(n_out, activation='sigmoid')(dense3)
    model = Model(inputs=[X_in, A_in], outputs=output)
    optimizer = Adam(lr=learning_rate)
    model.compile(optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['acc']) 
    model.summary()
    return model
# Read your csv file including the imaging id
X_set=pd.read_csv('../All_TLE_HS_MRneg_foroutcome99.csv')
for idx, row in X_set.iterrows():
    if len(row['id']) == 2:
       row['id'] = '0' + row['id']
       X_set.loc[idx, 'id'] = row['id']


Y_label=X_set.loc[:,['BOutcome']]
# set your label 0: good surgical outcome; 1: poor surgical outcome

# Read in cthick and pet data
train_data_cthick = np.array([pd.read_table("../mriall_1k_flip" + '/TLE_' + str(id) + '_cthick.txt',header=None, index_col=False).values for id in X_set['id']]).reshape(len(Y_label),1284)

edges1k=pd.read_table('edges1k.txt',names=['node1','node2'],encoding='UTF-8')
src, dst = tuple(zip(*np.array(edges1k)-1))
g = dgl.DGLGraph()
g.add_nodes(1284)
g.add_edges(src, dst)
g.add_edges(dst, src)
nx_G = g.to_networkx().to_undirected()
print('We have %d nodes.' % g.number_of_nodes())
print('We have %d edges.' % g.number_of_edges())
#print(label_all)
over_sample=SMOTE()
train_smote_data_all,train_smote_labels_all=over_sample.fit_resample(data_all,label_all)

L_Cam=LabTool.L_matrix(nx_G,train_smote_data_all.shape[0])
model_cam=modelBuilder(0.14,0.00035,16)
permutation = np.random.permutation(train_smote_data_all.shape[0])
shuffled_dataset = train_smote_data_all[permutation]  
shuffled_label=train_smote_labels_all[permutation]
#plotcurve(history_cam,' Augmented by Chebyshev menthod')
np.average(history_cam.history['val_acc'])

for kkk in range(1,901):
    over_sample=SMOTE()
    train_smote_data_all,train_smote_labels_all=over_sample.fit_resample(data_all,label_all)
    L_Cam=LabTool.L_matrix(nx_G,train_smote_data_all.shape[0])
    model_cam=modelBuilder(0.14,0.00035,16)
    permutation = np.random.permutation(train_smote_data_all.shape[0])
    shuffled_dataset = train_smote_data_all[permutation]  
    shuffled_label=train_smote_labels_all[permutation] 
    history_cam=model_cam.fit([shuffled_dataset,L_Cam],shuffled_label,batch_size=2,epochs=90,validation_split=0.2)

    print('the %d th iteration is done'%kkk)
