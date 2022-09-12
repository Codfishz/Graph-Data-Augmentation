import numpy
import torch
import torch.nn as nn
from torch.nn.functional import relu
import torch.nn.functional as F
from torch_geometric.nn.inits import reset
import dgl
from dgl.nn import GINConv
# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# GIN-based model for regression and classification on graphs.
# pylint: disable= no-member, arguments-differ, invalid-name
from dgl.nn.pytorch.glob import GlobalAttentionPooling, SumPooling, AvgPooling, MaxPooling, Set2Set

from conv import GIN

nn_act = torch.nn.ReLU()
F_act = F.relu
__all__ = ['GINPredictor']
class GINPredictor(nn.Module):
    def __init__(self,emb_dim=300,dropout=0.5,n_tasks=1,gamma=0.4):
        super(GINPredictor, self).__init__()
        self.dropout = dropout
        self.emb_dim = emb_dim
        self.num_tasks = n_tasks
        self.gamma  = gamma
        emb_dim_rat = emb_dim
        rationale_gnn_node =GIN(2,2,emb_dim_rat,emb_dim_rat*2,emb_dim_rat,0.5, False,"sum","sum")
        self.graph_encoder = GIN(5,2,emb_dim_rat,emb_dim_rat*2,emb_dim_rat,0.5, False,"sum","sum")
        print("2")
        rep_dim = emb_dim
        self.predictor = torch.nn.Sequential(torch.nn.Linear(rep_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), nn_act, torch.nn.Dropout(), torch.nn.Linear(2*emb_dim, self.num_tasks))
        print("3")
        self.separator = separator(
            rationale_gnn_node=rationale_gnn_node,
            gate_nn = torch.nn.Sequential(torch.nn.Linear(emb_dim_rat, 2*emb_dim_rat), torch.nn.BatchNorm1d(2*emb_dim_rat), torch.nn.ReLU(), torch.nn.Dropout(), torch.nn.Linear(2*emb_dim_rat, 1)),
            nn=None
            )
        print("4")
        self.predict = nn.Linear(emb_dim, n_tasks)
        print("5")

    def forward(self, g, h):
        """Graph-level regression/soft classification.
        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs
        categorical_node_feats : list of LongTensor of shape (N)
            * Input categorical node features
            * len(categorical_node_feats) should be the same as len(num_node_emb_list)
            * N is the total number of nodes in the batch of graphs
        categorical_edge_feats : list of LongTensor of shape (E)
            * Input categorical edge features
            * len(categorical_edge_feats) should be the same as
              len(num_edge_emb_list) in the arguments
            * E is the total number of edges in the batch of graphs
        Returns
        -------
        FloatTensor of shape (B, n_tasks)
            * Predictions on graphs
            * B for the number of graphs in the batch
        """
        h_node = self.graph_encoder(g,h)
        h_r, h_env, r_node_num, env_node_num = self.separator(g, h, h_node)
        h_rep = (h_r.unsqueeze(1) + h_env.unsqueeze(0)).view(-1, self.emb_dim)
        
        pred_rem = self.predictor(h_r)
        pred_rep = self.predictor(h_rep)

        loss_reg =  torch.abs(r_node_num / (r_node_num + env_node_num) - self.gamma  * torch.ones_like(r_node_num)).mean()
        loss_reg += (self.separator.non_zero_node_ratio - self.gamma  * torch.ones_like(r_node_num)).mean()

        output = {'pred_rep': pred_rep, 'pred_rem': pred_rem, 'loss_reg':loss_reg}
        return output
        
    def eval_forward(self,g,h):
        h_node = self.graph_encoder(g,h)
        h_r, _, _, _ = self.separator(g, h, h_node)
        pred_rem = self.predictor(h_r)
        return pred_rem

class separator(torch.nn.Module):
    def __init__(self, rationale_gnn_node, gate_nn, nn=None):
        super(separator, self).__init__()
        self.rationale_gnn_node = rationale_gnn_node
        self.gate_nn = gate_nn
        self.nn = nn
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.rationale_gnn_node)
        reset(self.gate_nn)
        reset(self.nn)

    def forward(self, g, h, h_node, size=None):
        x = self.rationale_gnn_node(g,h)
        
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        size = h.size(dim=0)

        gate = self.gate_nn(x).view(-1, 1)
        h_node = self.nn(h_node) if self.nn is not None else h_node
        assert gate.dim() == h_node.dim() and gate.size(0) == h_node.size(0)
        gate = torch.sigmoid(gate)

#        h_out = scatter_add(gate * h_node, h, dim=0, dim_size=size)
#        c_out = scatter_add((1 - gate) * h_node, h, dim=0, dim_size=size)
        h_out = torch.ones_like(1,size) *(gate * h_node)
        c_out = torch.ones_like(1,size) *((1 - gate) * h_node)

        r_node_num = scatter_add(gate, h, dim=0, dim_size=size)
        env_node_num = scatter_add((1 - gate), h, dim=0, dim_size=size)

#        non_zero_nodes = scatter_add((gate > 0).to(torch.float32), batch, dim=0, dim_size=size)
#        all_nodes = scatter_add(torch.ones_like(gate).to(torch.float32), batch, dim=0, dim_size=size)
        non_zero_nodes = scatter_add((gate > 0).to(torch.float32), h, dim=0, dim_size=size)
        all_nodes = scatter_add(torch.ones_like(gate).to(torch.float32), h, dim=0, dim_size=size)
        self.non_zero_node_ratio = non_zero_nodes / all_nodes

        return h_out, c_out, r_node_num + 1e-8 , env_node_num + 1e-8

