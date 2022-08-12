# some code are from https://github.com/HongtengXu/SGWB-Graphon

import copy
import cv2
import numpy as np
import torch

from skimage.restoration import denoise_tv_chambolle
from typing import List, Tuple


def graph_numpy2tensor(graphs: List[np.ndarray]) -> torch.Tensor:
    """
    Convert a list of np arrays to a pytorch tensor
    :param graphs: [K (N, N) adjacency matrices]
    :return:
        graph_tensor: [K, N, N] tensor
    """
    graph_tensor = np.array(graphs)
    return torch.from_numpy(graph_tensor).float()

def align_graphs((graphs: List[np.ndarray],padding: bool = False) -> Tuple[List[np.ndarray],List[np.ndarray], int, int]:)
    aligned_graphs = []
    normalized_node_degrees = []
    num_nodes=graphs[0].shape[0]
    
    for i in range(len(graphs)):
        node_degree = 0.5 * np.sum(graphs[i], axis=0) + 0.5 * np.sum(graphs[i], axis=1)
        node_degree /= np.sum(node_degree)
        
        idx = np.argsort(node_degree)  # ascending
        idx = idx[::-1]  # descending
        
        sorted_node_degree = node_degree[idx]
        sorted_node_degree = sorted_node_degree.reshape(-1, 1)
        sorted_graph = copy.deepcopy(graphs[i])
        sorted_graph = sorted_graph[idx, :]
        sorted_graph = sorted_graph[:, idx]
        
        normalized_node_degrees.append(sorted_node_degree)
        aligned_graphs.append(sorted_graph)

    return aligned_graphs, normalized_node_degrees, num_nodes, num_nodes
#def align_graphs(graphs: List[np.ndarray],
#                 padding: bool = False) -> Tuple[List[np.ndarray], List[np.ndarray], int, int]:
#    """
#    Align multiple graphs by sorting their nodes by descending node degrees
#    :param graphs: a list of binary adjacency matrices
#    :param padding: whether padding graphs to the same size or not
#    :return:
#        aligned_graphs: a list of aligned adjacency matrices
#        normalized_node_degrees: a list of sorted normalized node degrees (as node distributions)
#    """
#    num_nodes = [graphs[i].shape[0] for i in range(len(graphs))]
#    max_num = max(num_nodes)
#    min_num = min(num_nodes)
#
#    aligned_graphs = []
#    normalized_node_degrees = []
#    for i in range(len(graphs)):
#        num_i = graphs[i].shape[0]
#
#        node_degree = 0.5 * np.sum(graphs[i], axis=0) + 0.5 * np.sum(graphs[i], axis=1)
#        node_degree /= np.sum(node_degree)
#        idx = np.argsort(node_degree)  # ascending
#        idx = idx[::-1]  # descending
#
#
#        sorted_node_degree = node_degree[idx]
#        sorted_node_degree = sorted_node_degree.reshape(-1, 1)
#
#        sorted_graph = copy.deepcopy(graphs[i])
#        sorted_graph = sorted_graph[idx, :]
#        sorted_graph = sorted_graph[:, idx]
#
#        if padding:
#            # normalized_node_degree = np.ones((max_num, 1)) / max_num
#            normalized_node_degree = np.zeros((max_num, 1))
#            normalized_node_degree[:num_i, :] = sorted_node_degree
#            aligned_graph = np.zeros((max_num, max_num))
#            aligned_graph[:num_i, :num_i] = sorted_graph
#            normalized_node_degrees.append(normalized_node_degree)
#            aligned_graphs.append(aligned_graph)
#        else:
#            # normalized_node_degree = np.ones(sorted_node_degree.shape) / sorted_node_degree.shape[0]
#            # normalized_node_degrees.append(normalized_node_degree)
#            normalized_node_degrees.append(sorted_node_degree)
#            aligned_graphs.append(sorted_graph)
#
#    return aligned_graphs, normalized_node_degrees, max_num, min_num


def universal_svd(aligned_graphs: List[np.ndarray], threshold: float = 2.02) -> np.ndarray:
    """
    Estimate a graphon by universal singular value thresholding.
    Reference:
    Chatterjee, Sourav.
    "Matrix estimation by universal singular value thresholding."
    The Annals of Statistics 43.1 (2015): 177-214.
    :param aligned_graphs: a list of (N, N) adjacency matrices
    :param threshold: the threshold for singular values
    :return: graphon: the estimated (r, r) graphon model
    """
    aligned_graphs = graph_numpy2tensor(aligned_graphs)
    num_graphs = aligned_graphs.size(0)

    if num_graphs > 1:
        sum_graph = torch.mean(aligned_graphs, dim=0)
    else:
        sum_graph = aligned_graphs[0, :, :]  # (N, N)

    num_nodes = sum_graph.size(0)

    u, s, v = torch.svd(sum_graph)
    singular_threshold = threshold * (num_nodes ** 0.5)
    binary_s = torch.lt(s, singular_threshold)
    s[binary_s] = 0
    graphon = u @ torch.diag(s) @ torch.t(v)
    graphon[graphon > 1] = 1
    graphon[graphon < 0] = 0
    graphon = graphon.numpy()
    return graphon
