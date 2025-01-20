import argparse
import os
import sys

import dgl
import networkx as nx
import pandas as pd
import torch
import torch.nn.functional as F

from sklearn.preprocessing import MinMaxScaler
from torch_geometric.nn import GraphSAGE
from torch_geometric.utils.convert import from_dgl


def get_batch(data):
    ben_data = data[data.Label == 0].copy()
    mal_data = data[data.Label == 1].copy()

    while len(ben_data) >= 10000 and len(mal_data) >= 1000:
        ben_batch = ben_data.sample(10000)
        mal_batch = mal_data.sample(1000)

        ben_data = ben_data.drop(ben_batch.index)
        mal_data = mal_data.drop(mal_batch.index)

        batch = pd.concat([ben_batch, mal_batch], ignore_index=True).sample(frac=1)
        yield batch
    else:
        ben_batch = ben_data.copy()
        mal_batch = mal_data.copy()

        ben_data = ben_data.drop(ben_batch.index)
        mal_data = mal_data.drop(mal_batch.index)

        batch = pd.concat([ben_batch, mal_batch], ignore_index=True).sample(frac=1)
        yield batch


def to_graph(data):
    G = nx.from_pandas_edgelist(data, source='SrcAddrPort', target='DstAddrPort', edge_attr=['x', 'Label'], create_using=nx.MultiGraph())
    G = G.to_directed()

    g = dgl.from_networkx(G, edge_attrs=['x', 'Label'])
    g = g.line_graph(shared=True)

    return from_dgl(g)


parser = argparse.ArgumentParser(description='Train GraphSAGE model')
parser.add_argument('--train-data', type=str, required=True, help='path to train data')
parser.add_argument('--model', type=str, required=True, help='path to save the GraphSAGE model')

args = parser.parse_args()

if not os.path.exists(args.train_data) or not os.path.isfile(args.train_data):
    sys.exit('Path to train data does not exist or is not a file')

train_data = pd.read_csv(args.train_data)

feat = list(train_data)
feat.remove('SrcAddrPort')
feat.remove('DstAddrPort')
feat.remove('Label')

scaler = MinMaxScaler()
train_data[feat] = scaler.fit_transform(train_data[feat])

train_data.insert(42, 'x', train_data[feat].values.tolist())

model = GraphSAGE(
    in_channels=40,
    hidden_channels=64,
    num_layers=2,
    out_channels=2,
    dropout=0.2
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 100
for epoch in range(1, epochs + 1):
    model.train()

    total_loss = total_nodes = 0
    for batch in get_batch(train_data):
        graph = to_graph(batch)
        
        optimizer.zero_grad()
        out = model(graph.x, graph.edge_index)
        loss = F.cross_entropy(out, graph.Label)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * graph.num_nodes
        total_nodes += graph.num_nodes
    
    print('Epoch {:03d}, Loss = {:.3f}'.format(epoch, total_loss / total_nodes))

torch.save(model, args.model)