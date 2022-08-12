#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 11:42:31 2021

@author: guojiawei
"""

import tensorflow as tf
from sklearn.metrics import roc_curve,auc,classification_report
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from spektral.utils import normalized_laplacian
from keras.models import Model
def curveroc(ture, pre):
    fpr, tpr, thresholds=roc_curve(ture,pre)
    AUC = auc(fpr, tpr)
    plt.plot(fpr,tpr,marker = 'o',color='darkorange',label='ROC curve (area = %0.2f)'%AUC)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.show()
    print(AUC)
def plotcurve(history,a):
  plt.figure(1)
  fig, ax = plt.subplots(1,2,figsize = (16,4))
  ax[0].plot(history.history['loss'],color='#EFAEA4',label = 'Training Loss')
  ax[0].plot(history.history['val_loss'],color='#B2D7D0',label = 'Test Loss')
  ax[1].plot(history.history['acc'],color='#EFAEA4',label = 'Training Accuracy')
  ax[1].plot(history.history['val_acc'],color='#B2D7D0',label = 'Test Accuracy')
  ax[0].legend()
  ax[1].legend()
  ax[0].set_xlabel('Epochs')
  ax[1].set_xlabel('Epochs');
  ax[0].set_ylabel('Loss')
  ax[1].set_ylabel('Accuracy %');
  fig.suptitle('Training on GCN'+a, fontsize = 24)   
def getcross(idx):
    cross0=[]
    cross1=[]
    for i in range(1,len(idx)+1,1):

        for j in range(1,len(idx)+1,1):
            cross0.append(np.intersect1d(idx[i-1],idx[j-1]).shape[0])

        cross1.append(cross0)
        cross0=[]
    return cross1
def getplot(cross,z,label):
    z=z-1
    c=[]
    for i in range(1,len(label)+1,1):
         if(np.array(label)[i-1,0]==1):
                c.append('r')
         else:
                c.append('g')
    aa=range(1,len(cross)+1,1)
    fig, ax=plt.subplots(figsize=(8,8 ))
    plt.bar(aa,cross[z,:],label=label,color=c,tick_label=c)
    for a in aa:
       plt.text(a, 51,'%.0f' % a, ha='center', va='bottom', fontsize=10)
    plt.xticks(aa,size='medium',rotation=30)
    print('this data point is',label[z][0])
    print('the color is',c[z])
def preformanceEv(model,x,L,y,history):
    pre=model.predict([x,L])
    plt.figure(1)
    plotcurve(history,' cthick')
    plt.show()
    idx=pre>=0.5
    idx2=pre<0.5
    pre[idx]=1
    pre[idx2]=0
    plt.figure(2)
    curveroc(y,pre)
    plt.show()
    print(classification_report(y,pre))
def L_matrix(nx_G,num):
    A= nx.adjacency_matrix(nx_G)
    L=normalized_laplacian(A, symmetric=True)
    L1=L.todense()
    L=[]
    for i in range(num):
        L.append(L1)
    L=tf.stack(np.array(L))
    return L
def A_matrix(nx_G,num):
     A= nx.adjacency_matrix(nx_G)
     A1= A.todense()
     A=[]
     for i in range(num):
         A.append(A1)
     A=tf.stack(np.array(A))
     return A
def getNode(model,layer,x_in,L,numOfNode):
    #model.summary()
    last_layer=model.get_layer(layer)
    maps=Model(inputs=[model.inputs],outputs=[model.output,last_layer.output])
    #maps.summary()
    with tf.GradientTape() as tape:
         model_out,last_out=maps([x_in,L])
      #print(1)
    grads=tape.gradient(model_out,last_out)
    #print(model_out)
    #print(grads.shape)
    pooled_grads=tf.reduce_mean(grads,axis=1)
    #print(pooled_grads.shape)
    names = locals()
    heatmap=[]
    #print(last_out.shape)
    for i in range(1,x_in.shape[0]+1,1):
            names['heatmap'+str(i)]= tf.reduce_mean(tf.multiply(pooled_grads[i-1], last_out[i-1]), axis=-1)
            heatmap.append(names['heatmap' + str(i) ])
    
    heatmap_1=np.array(heatmap)
    #print(heatmap_1.shape)
    heatmap_1=heatmap_1.reshape(x_in.shape[0],1284)
    heatmap_all2=np.absolute(heatmap_1)
    names2=locals()
    idxall1=[]
    idxall2=[]
    for i in range(1,x_in.shape[0]+1,1):
        names2['idxall_'+str(i)]=heatmap_all2[i-1].argsort()[0:numOfNode]
        idxall2.append(names['idxall_'+str(i)])
    idxall2=np.array(idxall2)
    cross0=[]
    cross1=[]
    for i in range(1,x_in.shape[0]+1,1):
        for j in range(1,x_in.shape[0]+1,1):
            cross0.append(np.intersect1d(idxall2[i-1],idxall2[j-1]).shape[0])
        cross1.append(cross0)
        cross0=[]
    cross=np.array(cross1)
    idxall2=np.array(idxall2)
    return cross,idxall2,last_out