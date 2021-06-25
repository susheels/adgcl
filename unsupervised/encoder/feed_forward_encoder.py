import torch
from torch.nn import Sequential, Linear, ReLU, Dropout


class FeedForwardNetwork(torch.nn.Module):
	def __init__(self, in_features, out_features, num_fc_layers, dropout):
		super(FeedForwardNetwork, self).__init__()

		self.num_fc_layers = num_fc_layers
		self.fcs = torch.nn.ModuleList()

		for i in range(num_fc_layers):
			if i == num_fc_layers -1:
				fc = Linear(in_features, out_features)
			else:
				fc = Sequential(Linear(in_features, in_features), ReLU(), Dropout(p=dropout))
			self.fcs.append(fc)

	def forward(self, x):
		for i in range(self.num_fc_layers):
			x = self.fcs[i](x)
		return x

class EncoderFeedForward(torch.nn.Module):
	def __init__(self,  num_features, dim, num_gc_layers, num_fc_layers, out_features, dropout):
		super(EncoderFeedForward, self).__init__()

		self.encoder = Encoder(num_features, dim, num_gc_layers)
		input_size_to_feed_forward = dim*num_gc_layers
		self.feed_forward = FeedForwardNetwork(input_size_to_feed_forward, out_features, num_fc_layers, dropout)

	def forward(self, batch, x, edge_index, edge_weight=None):

		x = self.encoder(batch, x, edge_index, edge_weight)
		x = self.feed_forward(x)
		x = F.log_softmax(x, dim=-1)
		return x