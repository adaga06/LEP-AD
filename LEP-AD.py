import argparse
import torch
import torch.nn as nn
import pickle
import pandas as pd
import numpy as np
import os
import wandb
from train import train,predicting
from utils import calculate_metrics

from preprocess_data import DTADataset,collate
from sklearn.model_selection import train_test_split
# Create the parser object
parser = argparse.ArgumentParser(description='Example script for argparse')
# Add arguments
parser.add_argument('--dataset', type=str, default='davis', help='Dataset to use')
parser.add_argument('--batch-size', type=int, default=512, help='Batch size')
parser.add_argument('--output_dim', type=int, default=128, help='Output dimension')
parser.add_argument('--heads', type=int, default=2, help='Number of heads for multi-head attention')
parser.add_argument('--n_layers', type=int, default=3, help='Number of transformer layers')
parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer to use')
parser.add_argument('--scheduler', action='store_true', help='Whether to use a learning rate scheduler')
# Parse the arguments
args = parser.parse_args()
# Access the argument values
dataset = args.dataset
batch_size = args.batch_size
output_dim = args.output_dim
heads = args.heads
n_layers = args.n_layers
optimizer = args.optimizer
scheduler = args.scheduler
# Do something with the arguments
print(f'Dataset: {dataset}')
print(f'Batch size: {batch_size}')
print(f'Output dimension: {output_dim}')
print(f'Number of heads: {heads}')
print(f'Number of transformer layers: {n_layers}')
print(f'Optimizer: {optimizer}')
print(f'Use learning rate scheduler: {scheduler}')

def load_drug_graphs(dataset):
    with open('data/'+dataset+'/temp/smile_graph.pickle','rb') as f:
        smile_graph = pickle.load(f,fix_imports=True)
    return smile_graph


def load_target_rep(dataset):
    with open('data/'+dataset+'/temp/protein_rep.pickle', 'rb') as handle:
        target_reps_dict = pickle.load(handle,fix_imports=True)
    return target_reps_dict

def processed_data(dataset):
    smile_graph = load_drug_graphs(dataset)
    target_reps_dict = load_target_rep(dataset)

    return smile_graph,target_reps_dict

smile_graph,target_reps_dict = processed_data(dataset)


import warnings
warnings.filterwarnings("ignore")
df_train_fold = pd.read_csv('data/' + dataset + '/'+ dataset+'_' + 'train' + '.csv')
train_drugs, train_prot_keys, train_Y = list(df_train_fold['compound_iso_smiles']), list(df_train_fold['target_sequence']), list(df_train_fold['affinity'])
train_drugs, train_prot_keys, train_Y = np.asarray(train_drugs), np.asarray(train_prot_keys), np.asarray(train_Y)

train_dataset = DTADataset(root='data', dataset=dataset + '_' + 'train', xd=train_drugs, target_key=train_prot_keys,
                            y=train_Y, smile_graph=smile_graph, target_rep=target_reps_dict)

# print(train_dataset)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cuda_name = 'cuda:0'
print('cuda_name:', cuda_name)
cross_validation_flag = True

TRAIN_BATCH_SIZE = batch_size
TEST_BATCH_SIZE = batch_size
LR = 0.0005
NUM_EPOCHS = 3000

train_data,valid_data=train_test_split(train_dataset,shuffle=True,test_size=0.2)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True,num_workers=8,
                                            collate_fn=collate)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=TEST_BATCH_SIZE, shuffle=False,num_workers=4,
                                            collate_fn=collate)

from model import GNNNet



print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

models_dir = 'models'
results_dir = 'results'

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Main program: iterate over different datasets
result_str = ''
model = GNNNet()
model.to(device)

model_st = GNNNet.__name__
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


print('yes')
best_mse = 1000
best_test_mse = 1000
best_epoch = -1
# model_file_name = 'models/model_' + model_st + '_' + dataset + '_' + str(fold) + '.model'
wandb.init(project="LEP-AD-Ablation-study",
           name='dataset '+dataset+',batch '+str(batch_size)+',output_dim '+str(output_dim)+',heads '+str(heads),
            entity="daga06")
for epoch in range(NUM_EPOCHS):
    train(model, device, train_loader, optimizer, epoch + 1)
    print('predicting for valid data')
    G, P = predicting(model, device, valid_loader)
    val_mse,val_ci,val_rm2 = calculate_metrics(G, P, dataset)
    wandb.log({"val_ci": val_ci})
    wandb.log({"val_mse": val_mse})
    wandb.log({"val_rm2": val_rm2})
df_test_fold = pd.read_csv('data/' + dataset + '/'+ dataset+'_' + 'test' + '.csv')
test_drugs, test_prot_keys, test_Y = list(df_test_fold['compound_iso_smiles']), list(df_test_fold['target_sequence']), list(df_test_fold['affinity'])
test_drugs, test_prot_keys, test_Y = np.asarray(test_drugs), np.asarray(test_prot_keys), np.asarray(test_Y)

test_data = DTADataset(root='data', dataset=dataset + '_' + 'test', xd=test_drugs, target_key=test_prot_keys,
                            y=test_Y, smile_graph=smile_graph, target_rep=target_reps_dict)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False,num_workers=4,
                                            collate_fn=collate)
