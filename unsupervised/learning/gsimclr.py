import torch
from torch.nn import Sequential, Linear, ReLU

class GSimCLR(torch.nn.Module):
	def __init__(self, encoder, proj_hidden_dim=300):
		super(GSimCLR, self).__init__()

		self.encoder = encoder
		self.input_proj_dim = self.encoder.out_graph_dim

		self.proj_head = Sequential(Linear(self.input_proj_dim, proj_hidden_dim), ReLU(inplace=True),
		                            Linear(proj_hidden_dim, proj_hidden_dim))

		self.init_emb()

	def init_emb(self):
		for m in self.modules():
			if isinstance(m, Linear):
				torch.nn.init.xavier_uniform_(m.weight.data)
				if m.bias is not None:
					m.bias.data.fill_(0.0)

	def forward(self, batch, x, edge_index, edge_attr=None, edge_weight=None):
		if edge_attr is not None:
			z, _ = self.encoder(batch, x, edge_index, edge_attr, edge_weight)
		else:
			z, _ = self.encoder(batch, x, edge_index, edge_weight)

		z = self.proj_head(z)
		# z shape -> Batch x (hidden_dim*num_gc_layers)
		return z

	@staticmethod
	def calc_loss(x, x_aug, temperature=0.2):
		# x shape -> Batch x (hidden_dim*num_gc_layers)

		batch_size, _ = x.size()
		x_abs = x.norm(dim=1)
		x_aug_abs = x_aug.norm(dim=1)

		sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
		sim_matrix = torch.exp(sim_matrix / temperature)
		pos_sim = sim_matrix[range(batch_size), range(batch_size)]
		loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
		loss = - torch.log(loss).mean()

		return loss