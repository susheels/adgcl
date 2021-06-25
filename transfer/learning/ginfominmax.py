import torch
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import global_mean_pool



class GInfoMinMax(torch.nn.Module):
    def __init__(self, gnn, proj_hidden_dim=300):
        super(GInfoMinMax, self).__init__()

        self.gnn = gnn
        self.pool = global_mean_pool
        self.input_proj_dim = gnn.emb_dim

        self.projection_head = Sequential(Linear(self.input_proj_dim, proj_hidden_dim), ReLU(inplace=True),
                                       Linear(proj_hidden_dim, proj_hidden_dim))

        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, batch, x, edge_index, edge_attr, edge_weight=None):
        x = self.gnn(x, edge_index, edge_attr, edge_weight)
        x = self.pool(x, batch)
        x = self.projection_head(x)
        return x

    @staticmethod
    def calc_loss( x, x_aug, temperature=0.2, sym=False):
        # x and x_aug shape -> Batch x proj_hidden_dim

        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / temperature)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        if sym:

            loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim)
            loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)

            loss_0 = - torch.log(loss_0).mean()
            loss_1 = - torch.log(loss_1).mean()
            loss = (loss_0 + loss_1)/2.0
        else:
            loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
            loss = - torch.log(loss).mean()

        return loss


