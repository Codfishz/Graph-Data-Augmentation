import torch
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import dgl
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

if __name__ == "__main__":
#    u, v = torch.tensor([0, 0, 0, 1]), torch.tensor([1, 2, 3, 3])
#    g = dgl.graph((u, v))
#    with g.local_scope():
#        print(g.is_multigraph)
    
    device = torch.device("cuda:" + str(0)) if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    dataset = MRIDataset()
    nx_G=dataset.sampleGraph
    dataset_size=len(dataset)
    print(nx_G.is_multigraph)

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
    for step, batch in enumerate(train_loader):
        batch_x = batch[0].to(device)
        batch_y = batch[1].to(device)
        print(batch_y.size())

    evaluator = Evaluator()

    n_train_data, n_val_data, n_test_data = len(train_loader.dataset), len(valid_loader.dataset), len(test_loader.dataset)
    print(f"# Train: {n_train_data}  #Test: {n_test_data} #Val: {n_val_data}")

    model = GINPredictor(emb_dim=256,dropout=0.5,n_tasks=1,gamma=0.4).to(device)
    init_weights(model, 'normal', init_gain=0.02)
    opt_separator = optim.Adam(model.separator.parameters(), lr=0.01, weight_decay=5e-6)
    opt_predictor = optim.Adam(list(model.graph_encoder.parameters())+list(model.predict.parameters()), lr=0.01, weight_decay=5e-6)
    optimizers = {'separator': opt_separator, 'predictor': opt_predictor}

    schedulers = None
    cnt_wait = 0
    best_epoch = 0
    path_list=[1,4]
    for epoch in range(10):
        print("=====Epoch {}".format(epoch))
        path = epoch % int(path_list[-1])
        if path in list(range(int(path_list[0]))):
            optimizer_name = 'separator'
        elif path in list(range(int(path_list[0]), int(path_list[1]))):
            optimizer_name = 'predictor'

        train(model, device, train_loader,nx_G, optimizers, "classification", optimizer_name)

        if schedulers != None:
            schedulers[optimizer_name].step()
        train_perf = eval(model, device, train_loader, nx_G, evaluator)[0]
        valid_perf = eval(model, device, valid_loader, nx_G, evaluator)[0]
        update_test = False
        if epoch != 0:
            if valid_perf >  best_valid_perf:
                update_test = True
        if update_test or epoch == 0:
            best_valid_perf = valid_perf
            cnt_wait = 0
            best_epoch = epoch
            test_perfs = eval(model, device, test_loader, evaluator)
            test_auc  = test_perfs[0]
            print({'Metric': 'AUC', 'Train': train_perf, 'Validation': valid_perf, 'Test': test_auc})
        else:
            print({'Train': train_perf, 'Validation': valid_perf})
            cnt_wait += 1
    print('Finished training! Results from epoch {} with best validation {}.'.format(best_epoch, best_valid_perf))
    print('Test auc: {}'.format(test_auc))
