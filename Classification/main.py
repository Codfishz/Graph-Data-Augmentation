import torch
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

import numpy as np
from tqdm import tqdm

## dataset
#from sklearn.model_selection import train_test_split
#from dataset import PolymerRegDataset
import Datasets
from Datasets import Evaluator, MRIDataset
#from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

## training
from models import GINPredictor
from utils import init_weights, get_args, train, eval


def main(args):
    print(args)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    dataset = MRIDataset()
    dataset_size=len(dataset)
    num_node_list=np.empty(dataset_size,dtype=np.int64)
    num_node_list.fill(int(dataset.sampleGraph.number_of_nodes()))
    num_edge_list=np.empty(dataset_size,dtype=np.int64)
    num_edge_list.fill(int(dataset.sampleGraph.number_of_edges()))
    
    
    train_dataset_size=int(dataset_size*0.6)
    valid_dataset_size=int(dataset_size*0.2)
    test_dataset_size=dataset_size-train_dataset_size-valid_dataset_size
    
    train_dataset, t_v_dataset = torch.utils.data.random_split(dataset, [train_dataset_size,valid_dataset_size+test_dataset_size])
    valid_dataset,test_dataset = torch.utils.data.random_split(t_v_dataset, [valid_dataset_size,test_dataset_size])
#    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = 0)
#    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers = 0)
#    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers = 0)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers = 0)
    valid_loader = DataLoader(valid_dataset, batch_size=256, shuffle=False, num_workers = 0)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers = 0)
    
    evaluator = Evaluator()

    
    n_train_data, n_val_data, n_test_data = len(train_loader.dataset), len(valid_loader.dataset), len(test_loader.dataset)
    print(f"# Train: {n_train_data}  #Test: {n_test_data} #Val: {n_val_data}")

#    model = GINPredictor(num_node_emb_list = num_node_list, num_edge_emb_list = num_edge_list, num_layers=5,                 emb_dim=args.emb_dim, JK='last', dropout=args.drop_ratio, readout='mean',n_tasks=1,gamma=args.gamma).to(device)
    
    model = GINPredictor(num_node_emb_list = num_node_list, num_edge_emb_list = num_edge_list, num_layers=5,                 emb_dim=128, JK='last', dropout=0.5, readout='mean',n_tasks=1,gamma=0.4).to(device)
    init_weights(model, 'normal', init_gain=0.02)
    opt_separator = optim.Adam(model.separator.parameters(), lr=args.lr, weight_decay=args.l2reg)
    opt_predictor = optim.Adam(list(model.gnn.parameters())+list(model.predictor.parameters()), lr=args.lr, weight_decay=args.l2reg)
    optimizers = {'separator': opt_separator, 'predictor': opt_predictor}
    if args.use_lr_scheduler:
        schedulers = {}
        for opt_name, opt in optimizers.items():
            schedulers[opt_name] = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100, eta_min=1e-4)
    else:
        schedulers = None
    cnt_wait = 0
    best_epoch = 0
    for epoch in range(args.epochs):
        print("=====Epoch {}".format(epoch))
        path = epoch % int(args.path_list[-1])
        if path in list(range(int(args.path_list[0]))):
            optimizer_name = 'separator'
        elif path in list(range(int(args.path_list[0]), int(args.path_list[1]))):
            optimizer_name = 'predictor'

        train(args, model, device, train_loader, optimizers, "classification", optimizer_name)

        if schedulers != None:
            schedulers[optimizer_name].step()
        train_perf = eval(args, model, device, train_loader, evaluator)[0]
        valid_perf = eval(args, model, device, valid_loader, evaluator)[0]
        update_test = False
        if epoch != 0:
            if valid_perf >  best_valid_perf:
                update_test = True
        if update_test or epoch == 0:
            best_valid_perf = valid_perf
            cnt_wait = 0
            best_epoch = epoch
            test_perfs = eval(args, model, device, test_loader, evaluator)
            test_auc  = test_perfs[0]
            print({'Metric': 'AUC', 'Train': train_perf, 'Validation': valid_perf, 'Test': test_auc})
        else:
            print({'Train': train_perf, 'Validation': valid_perf})
            cnt_wait += 1
            if cnt_wait > args.patience:
                break
    print('Finished training! Results from epoch {} with best validation {}.'.format(best_epoch, best_valid_perf))
    print('Test auc: {}'.format(test_auc))
    return [best_valid_perf, test_auc]

if __name__ == "__main__":
    args = get_args()
    valid_auc, test_auc = main(args)
    print('valid_auc:'+valid_auc)
    print('test_auc:'+test_auc)
