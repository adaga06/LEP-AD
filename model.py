import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import TransformerConv,GATConv, GCNConv,global_max_pool as gmp, global_add_pool as gap,global_mean_pool as gep,global_sort_pool
from torch_geometric.utils import dropout_adj



# GCN based model
class GNNNet(torch.nn.Module):
    def __init__(self, n_output=1, num_features_mol=78, output_dim=128, dropout=0.2):
        super(GNNNet, self).__init__()

        print('GNNNet Loaded')
        self.n_output = n_output
        self.mol_conv1 = TransformerConv(num_features_mol, num_features_mol,heads=4)
        self.mol_conv2 = TransformerConv(4*num_features_mol, num_features_mol * 2,heads=2)
        self.mol_conv3 = TransformerConv(num_features_mol * 4, num_features_mol * 4,heads=1)
        self.mol_fc_g1 = torch.nn.Linear(num_features_mol * 4, 1024)
        self.mol_fc_g2 = torch.nn.Linear(1024, output_dim)

        # self.pro_conv1 = GCNConv(embed_dim, embed_dim)
        # self.pro_conv1 = GCNConv(num_features_pro, num_features_pro)

        # self.pro_conv4 = GCNConv(embed_dim * 4, embed_dim * 8)
        # self.pro_fc_g1 = torch.nn.Linear(num_features_pro, 1024)
        # self.pro_fc_g2 = torch.nn.Linear(1024, output_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # combined layers
        self.fc1 = nn.Linear(2688, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)

    def forward(self, data_mol, data_pro):
        # get graph input
        mol_x, mol_edge_index, mol_batch = data_mol.x, data_mol.edge_index, data_mol.batch
        # get protein input
        # target_x, target_edge_index, target_batch = data_pro.x, data_pro.edge_index, data_pro.batch

        # target_seq=data_pro.target

        # print('size')
        # print('mol_x', mol_x.size(), 'edge_index', mol_edge_index.size(), 'batch',mol_batch, mol_batch.size())
        # print('target_x', target_x.size(), 'target_edge_index', target_edge_index,target_edge_index.size(), 'batch',target_batch, target_batch.size())

        x = self.mol_conv1(mol_x, mol_edge_index)
        x = self.relu(x)
        # print(x.shape)

        # mol_edge_index, _ = dropout_adj(mol_edge_index, training=self.training)
        x = self.mol_conv2(x, mol_edge_index)
        x = self.relu(x)
        # print(x.shape)

        # mol_edge_index, _ = dropout_adj(mol_edge_index, training=self.training)
        x = self.mol_conv3(x, mol_edge_index)
        x = self.relu(x)
        # print(x.shape)

        x = gep(x, mol_batch)  # global pooling

        # print(x.shape)
        # flatten
        x = self.relu(self.mol_fc_g1(x))
        x = self.dropout(x)
        x = self.mol_fc_g2(x)
        x = self.dropout(x)

        # # xt = self.pro_conv1(target_x, target_edge_index)
        # xt = self.relu(xt)
        # xt = gep(xt, target_batch)  # global pooling

        # # flatten
        # xt = self.relu(self.pro_fc_g1(xt))
        # xt = self.dropout(xt)
        # xt = self.pro_fc_g2(xt)
        # xt = self.dropout(xt)

        # print(x.size(), xt.size())
        # concat
        xc = torch.cat((x, data_pro), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out

