import torch
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import global_mean_pool

class GSimCLR(torch.nn.Module):

    def __init__(self, gnn):
        super(GSimCLR, self).__init__()
        self.gnn = gnn
        self.pool = global_mean_pool
        self.projection_head = Sequential(Linear(300, 300), ReLU(inplace=True), Linear(300, 300))

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.gnn(x, edge_index, edge_attr)
        x = self.pool(x, batch)
        x = self.projection_head(x)
        return x

    @staticmethod
    def calc_loss(x1, x2):
        T = 0.1
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()
        return loss