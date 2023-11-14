import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch_geometric
from torch_geometric.data import Data
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch_geometric.nn import GCNConv, GATConv


class dataset_vibgnn:
    def __init__(self, edge_list, classes, features=None, twohop=False, verbose=True):

        self.neg_examples = []
        self.n_nodes = len(classes)
        self.edge_list = edge_list

        if verbose:
            print("Building the graph")

        if features is None:
            x = torch.tensor(np.identity(self.n_nodes), dtype=torch.float)
        else:
            x = torch.tensor(features, dtype=torch.float)

        unzipped_el = list(zip(*edge_list))
        edge_index = torch.tensor([unzipped_el[0], unzipped_el[1]], dtype=torch.long)

        print("graph")
        self.graph = Data(x=x, edge_index=edge_index, y=torch.tensor(classes))

        if verbose:
            print(f'Number of features: {self.graph.num_features}')
            print(f'Number of nodes: {self.graph.num_nodes}')
            print(f'Number of edges: {self.graph.num_edges}')
            print(f'Average node degree: {self.graph.num_edges / self.graph.num_nodes:.2f}')
            print(f'Has isolated nodes: {self.graph.has_isolated_nodes()}')
            print(f'Has self-loops: {self.graph.has_self_loops()}')
            print(f'Is undirected: {self.graph.is_undirected()}')
            print("Building the same class matrix")

        # Building pairwise matrix that checks if nodes are in the same class
        # Very dirty, could be optimized

        n_classes = len(list(set(classes)))
        # onehot_classes = np.zeros((self.n_nodes,n_classes))
        # for i in range(self.n_nodes):
        #  onehot_classes[i,classes[i]] = 1
        onehot_classes = F.one_hot(torch.as_tensor(classes).long(), num_classes=n_classes)
        same_class = 1 - onehot_classes @ onehot_classes.T
        self.same_class = same_class

        # Building positive examples
        # Dirty
        #
        if twohop:
            print("To Do")
        else:
            self.pos_examples = [(i[0], i[1], 1, same_class[i[0], i[1]]) for i in edge_list]

    def build_train(self):
        graph_train = copy.deepcopy(self.graph)
        unzipped_el = list(zip(*self.edge_list))
        edge_index = torch.tensor([unzipped_el[0], unzipped_el[1]], dtype=torch.long)
        graph_train.edge_index = edge_index
        return graph_train


class GAT(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, heads=2, num_layers=2):
        super().__init__()
        layers = []
        in_channel, out_channel = num_features, hidden_channels[0]
        n_h = 1
        for l in range(num_layers-1):
            layers += [GATConv(in_channel*n_h, out_channel, heads=heads, concat=True), nn.ReLU(inplace=True),
                       nn.Dropout(0.6)]
            in_channel = out_channel
            out_channel = int(out_channel/2)
            print(in_channel, out_channel)
            n_h = heads

        layers += [GATConv(hidden_channels[0]*heads, hidden_channels[1], heads=heads, concat=False)]
        #layers += [GATConv(in_channel * heads, out_channel, heads=heads, concat=False)]
        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index):
        for l in self.layers:
            if isinstance(l, torch_geometric.nn.GATConv):
                x = l(x, edge_index)
            else:
                x = l(x)
        return x


class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_layers=2):
        super().__init__()

        layers = []
        in_channel, out_channel = num_features, hidden_channels[0]

        for l in range(num_layers-1):
            layers += [GCNConv(in_channel, out_channel), nn.ReLU(inplace=True),
                       nn.Dropout(0.6)]
            in_channel = out_channel
            out_channel = int(out_channel/2)

        layers += [GCNConv(in_channel, out_channel)]

        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index):
        for l in self.layers:
            if isinstance(l, torch_geometric.nn.GCNConv):
                x = l(x, edge_index)
            else:
                x = l(x)
        return x


class mlp(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(mlp, self).__init__()
        self.do = torch.nn.Dropout(p=0.2)
        self.fc1 = torch.nn.Linear(input_size, input_size)
        self.tanh = torch.nn.Tanh()
        self.fc2 = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.fc1(self.do(x))
        x = self.tanh(x)
        x = self.fc2(self.do(x))
        return x


class mlp_mv(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(mlp_mv, self).__init__()
        self.do = torch.nn.Dropout(p=0.2)
        self.fc1 = torch.nn.Linear(input_size, input_size)
        self.tanh = torch.nn.Tanh()
        self.fc2 = torch.nn.Linear(input_size, output_size)

    def forward(self, x_1, x_2, A):
        
        x_1 = torch.squeeze(x_1).long()
        x_2 = torch.squeeze(x_2).long()
        
        x_1 = A[x_1]
        x_2 = A[x_2]
        
        x_1 = self.fc1(self.do(x_1))
        x_1 = self.tanh(x_1)
        x_1 = self.fc2(self.do(x_1))
        
        x_2 = self.fc1(self.do(x_2))
        x_2 = self.tanh(x_2)
        x_2 = self.fc2(self.do(x_2))
        return x_1, x_2


class VIB(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, h=1, n_layers=2, encoder="GCN"):
        super().__init__()
        if encoder == "GAT":
            self.gnn = GAT(num_features, hidden_channels, heads=h, num_layers=n_layers)
        elif encoder == "GCN":
            self.gnn = GCN(num_features, hidden_channels, num_layers=n_layers)
        elif encoder == "MLP":
            self.encoder = mlp_mv(hidden_channels[0], hidden_channels[1])
        else:
            print("unknown encoder, using GCN")

        self.log_a = torch.nn.Parameter(torch.Tensor([0]))
        self.log_a_f = torch.nn.Parameter(torch.Tensor([0]))

        self.b = torch.nn.Parameter(torch.rand(1))
        self.b_f = torch.nn.Parameter(torch.rand(1))

        self.mlp_mu = mlp(hidden_channels[1], hidden_channels[1])
        self.mlp_logsigma = mlp(hidden_channels[1], hidden_channels[1])

        self.N = torch.distributions.Normal(0, 1)


    def distance(self, graph, edge_index, x_1, x_2, A=None):
        if A is None:
            
            embedding = self.gnn(graph, edge_index)

            x_1 = torch.squeeze(x_1).long()
            x_2 = torch.squeeze(x_2).long()

            dist = torch.sum(torch.square(embedding[x_1] - embedding[x_2]), 1)

        else:
            
            x_1, x_2 = self.encoder(x_1, x_2, A)

            dist = torch.sum(torch.square(x_1 - x_2), 1)

        return dist

    def forward(self, graph, edge_index, x_1, x_2, A=None):
        
        if A is None:
            embedding = self.gnn(graph, edge_index)

            x_1 = torch.squeeze(x_1).long()
            x_2 = torch.squeeze(x_2).long()

            dist = torch.sum(torch.square(embedding[x_1] - embedding[x_2]), 1)

            proba_p = torch.sigmoid(- torch.exp(self.log_a) * dist + self.b)
            proba_f = torch.sigmoid(- torch.exp(self.log_a_f) * dist + self.b_f)

            return proba_p, proba_f
        else:
            x_1, x_2 = self.encoder(x_1, x_2, A)

            dist = torch.sum(torch.square(x_1 - x_2), 1)

            proba_p = torch.sigmoid(- torch.exp(self.log_a) * dist + self.b)
            proba_f = torch.sigmoid(- torch.exp(self.log_a_f) * dist + self.b_f)

            return proba_p, proba_f
                    
    def loss(self, graph, edge_index, x_1, x_2, y, s, criterion, alpha,A=None):

        if A is None:
            embedding = self.gnn(graph, edge_index)

            x_1 = torch.squeeze(x_1).long()
            x_2 = torch.squeeze(x_2).long()

            dist = torch.sum(torch.square(embedding[x_1] - embedding[x_2]), 1)
        else:
            x_1,x_2 = self.encoder(x_1, x_2, A)
        
            dist = torch.sum(torch.square(x_1 - x_2), 1)
        
        
        proba_p = torch.sigmoid(- torch.exp(self.log_a) * dist + self.b)
        proba_f = torch.sigmoid(- torch.exp(self.log_a_f) * dist + self.b_f)

        loss_p = criterion(proba_p, torch.squeeze(y))  # Compute the loss solely based on the training nodes.
        loss_f = criterion(proba_f, torch.squeeze(s))

        loss = loss_p + alpha * loss_f

        return loss, loss_p, loss_f

    def loss_vib(self, graph, edge_index, x_1, x_2, y, s, criterion, alpha, beta=0, L=5,A=None):
    
        if A is None:
            embedding = self.gnn(graph, edge_index)

            x_1 = embedding[torch.squeeze(x_1).long()]
            x_2 = embedding[torch.squeeze(x_2).long()]
        else:
            x_1, x_2 = self.encoder(x_1, x_2, A)

        loss_p = 0
        loss_f = 0
        loss_prior = 0

        for i in range(L):
            mu_1 = self.mlp_mu(x_1)
            mu_2 = self.mlp_mu(x_2)

            sigma_1 = torch.exp(self.mlp_logsigma(x_1))
            sigma_2 = torch.exp(self.mlp_logsigma(x_2))

            z_1 = mu_1 + 0.0001 * sigma_1 * self.N.sample(mu_1.shape)
            z_2 = mu_2 + 0.0001 * sigma_2 * self.N.sample(mu_2.shape)

            dist = torch.sum(torch.square(z_1 - z_2), 1)

            proba_p = torch.sigmoid(- torch.exp(self.log_a) * dist + self.b)
            proba_f = torch.sigmoid(- torch.exp(self.log_a_f) * dist + self.b_f)

            loss_p += criterion(proba_p, torch.squeeze(y))  # Compute the loss solely based on the training nodes.
            loss_f += criterion(proba_f, torch.squeeze(s))

            loss_prior += (sigma_1 ** 2 + mu_1 ** 2 - torch.log(sigma_1) - 1).sum()  # à verifier
            loss_prior += (sigma_2 ** 2 + mu_1 ** 2 - torch.log(sigma_2) - 1).sum()  # à verifier

        loss_p /= L
        loss_f /= L
        loss_prior /= L

        loss = loss_p + alpha * loss_f + beta * loss_prior

        return loss, loss_p, loss_f


def visualize(model, dataset, color=None):
    with torch.no_grad():
        model.eval()
        h = model.gnn(dataset.graph.x, dataset.graph.edge_index)
        z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

        plt.figure(figsize=(10, 10))
        plt.xticks([])
        plt.yticks([])

        if color is None:
            color = dataset.graph.y
        plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
        plt.show()
