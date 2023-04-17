from collections import OrderedDict
import json,pickle
import networkx as nx
import numpy as np
import torch
import pandas as pd


from rdkit import RDLogger
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
RDLogger.DisableLog('rdApp.*')

def make_graphs(dataset):
    # load dataset
    dataset_path = 'data/' + dataset + '/'

    ligands = json.load(open(dataset_path + 'ligands_can.txt'), object_pairs_hook=OrderedDict)
    proteins = json.load(open(dataset_path + 'proteins.txt'), object_pairs_hook=OrderedDict)


    drugs = []
    drug_smiles = []
    # smiles
    for d in ligands.keys():
        lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]), isomericSmiles=True)
        drugs.append(lg)
        drug_smiles.append(ligands[d])

# one ont encoding
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        # print(x)
        raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    '''Maps inputs not in the allowable set to the last element.'''
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))
    
# mol atom feature for mol graph
def atom_features(atom):
    # 44 +11 +11 +11 +1
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'X']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])

# mol smile to mol graph edge index

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)

    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    mol_edge_weight=[]
    mol_adj = np.zeros((c_size, c_size))
    for e1, e2 in g.edges:
        mol_adj[e1, e2] = 1
        # edge_index.append([e1, e2])
    mol_adj += np.matrix(np.eye(mol_adj.shape[0]))
    index_row, index_col = np.where(mol_adj >= 0.5)
    for i, j in zip(index_row, index_col):
        edge_index.append([i, j])
        mol_edge_weight.append([1])
    # print('smile_to_graph')
    # print(np.array(features).shape)
    return c_size, features, edge_index,mol_edge_weight

# 14min
# compound_iso_smiles = drugs
def load_drug_graphs(dataset):
    # create smile graph
    # smile_graph = {}
    # i = 0
    # for smile in compound_iso_smiles:
    #     g = smile_to_graph(smile)
    #     smile_graph[smile] = g
    #     if i%1000==0:
    #         print(i)
    #     i+=1
    # with open('data/davis/temp/smile_graph.pickle','wb') as handle:
    #     pickle.dump(smile_graph, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('data/'+dataset+'/temp/smile_graph.pickle','rb') as f:
        smile_graph = pickle.load(f,fix_imports=True)
    return smile_graph


def load_target_rep(dataset):
    # esm_emb_path = 'data/' + dataset + '/protein_emb/UniRef50_'
    # protein_dict = json.load(open('data/davis/proteins.txt'))
    # target_reps_dict = {}
    # i=0
    # for key in proteins.keys():
    #     target_reps_dict[protein_dict[key]] = torch. load(esm_emb_path+ key + '.pt')['mean_representations'][36]
    #     if i%1000==0:
    #         print(i)
    #     i+=1

    # with open('data/davis/temp/protein_rep.pickle','wb') as handle:
    #     pickle.dump(target_reps_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('data/'+dataset+'/temp/protein_rep.pickle', 'rb') as handle:
        target_reps_dict = pickle.load(handle,fix_imports=True)
    return target_reps_dict

def processed_data(dataset):
    smile_graph = load_drug_graphs(dataset)
    target_reps_dict = load_target_rep(dataset)

    return smile_graph,target_reps_dict

import os
from torch_geometric.data import InMemoryDataset, DataLoader, Batch
from torch_geometric import data as DATA
import torch
import numpy as np
import torchvision.transforms as T

# initialize the dataset
class DTADataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='davis',
                 xd=None, y=None, transform= None,
                 pre_transform=None, smile_graph=None, target_key=None, target_rep=None):
        super(DTADataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        self.process(xd, target_key, y, smile_graph, target_rep)

    @property
    def raw_file_names(self):
        pass
        # return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + '_data_mol.pt', self.dataset + '_data_pro.pt']

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, xd, target_key, y, smile_graph, target_rep):
        assert (len(xd) == len(target_key) and len(xd) == len(y)), 'The three lists must be the same length!'
        data_list_mol = []
        data_list_pro = []
        data_len = len(xd)
        for i in range(data_len):
            entity1 = xd[i]
            # print(torch.Tensor(target_rep[target_key[i]][36]))
            # print(torch.FloatTensor(y[i]).shape)
            # torch.from_numpy
            labels = y[i]
            # print(labels,torch.FloatTensor([labels]),torch.FloatTensor([labels]).shape)
            
            # labels = torch.concat((torch.FloatTensor([labels]),torch.Tensor(target_rep[target_key[i]][36])))
            # print(labels)
            # print(labels.shape)
            # print('DTI')
            # convert SMILES to molecular representation using rdkit
            if entity1 in smile_graph.keys():
                c_size, features, edge_index,edge_weight = smile_graph[entity1]
            else:
                # print('graph not found')
                c_size, features, edge_index,edge_weight = smile_to_graph(entity1)
                # print('complete')
            # print(target_features.shape, target_edge_index.shape)
            # make the graph ready for PyTorch Geometrics GCN algorithms:
            GCNData_mol = DATA.Data(x=torch.Tensor(np.array(features)),
                                    edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                    y=torch.FloatTensor([labels])
                                    )
            GCNData_mol.__setitem__('c_size', torch.LongTensor([c_size]))
            data_list_mol.append(GCNData_mol)
            # print(i)
            # print(target_key[i])
            # print(target_rep)
            # print(target_rep[target_key[i]])
            data_list_pro.append(torch.Tensor(target_rep[target_key[i]]))
            if i%10000==0:
                print(i)
            # print(data_list_mol,data_list_pro)
   
            
        if self.pre_filter is not None:
            data_list_mol = [data for data in data_list_mol if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list_mol = [self.pre_transform(data) for data in data_list_mol]
        self.data_mol = data_list_mol
        self.data_pro = data_list_pro

    def __len__(self):
        return len(self.data_mol)

    def __getitem__(self, idx):
        return self.data_mol[idx], self.data_pro[idx]
        
def collate(batch):
    graphs = Batch.from_data_list([item[0] for item in batch])
    tensors = [item[1] for item in batch]
    tensors = torch.stack(tensors)

    return graphs,tensors