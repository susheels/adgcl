import math

import torch
import torch.nn.functional as F
from torch import nn


def log_sum_exp(x, axis=None):
	"""Log sum exp function
	Args:
		x: Input.
		axis: Axis over which to perform sum.
	Returns:
		torch.Tensor: log sum exp
	"""
	x_max = torch.max(x, axis)[0]
	y = torch.log((torch.exp(x - x_max)).sum(axis)) + x_max
	return y
def raise_measure_error(measure):
	supported_measures = ['GAN', 'JSD', 'X2', 'KL', 'RKL', 'DV', 'H2', 'W1']
	raise NotImplementedError(
		'Measure `{}` not supported. Supported: {}'.format(measure,
														   supported_measures))

def get_positive_expectation(p_samples, measure, average=True):
	"""Computes the positive part of a divergence / difference.
	Args:
		p_samples: Positive samples.
		measure: Measure to compute for.
		average: Average the result over samples.
	Returns:
		torch.Tensor
	"""
	log_2 = math.log(2.)

	if measure == 'GAN':
		Ep = - F.softplus(-p_samples)
	elif measure == 'JSD':
		Ep = log_2 - F.softplus(- p_samples)
	elif measure == 'X2':
		Ep = p_samples ** 2
	elif measure == 'KL':
		Ep = p_samples + 1.
	elif measure == 'RKL':
		Ep = -torch.exp(-p_samples)
	elif measure == 'DV':
		Ep = p_samples
	elif measure == 'H2':
		Ep = 1. - torch.exp(-p_samples)
	elif measure == 'W1':
		Ep = p_samples
	else:
		raise_measure_error(measure)

	if average:
		return Ep.mean()
	else:
		return Ep


def get_negative_expectation(q_samples, measure, average=True):
	"""Computes the negative part of a divergence / difference.
	Args:
		q_samples: Negative samples.
		measure: Measure to compute for.
		average: Average the result over samples.
	Returns:
		torch.Tensor
	"""
	log_2 = math.log(2.)

	if measure == 'GAN':
		Eq = F.softplus(-q_samples) + q_samples
	elif measure == 'JSD':
		Eq = F.softplus(-q_samples) + q_samples - log_2
	elif measure == 'X2':
		Eq = -0.5 * ((torch.sqrt(q_samples ** 2) + 1.) ** 2)
	elif measure == 'KL':
		Eq = torch.exp(q_samples)
	elif measure == 'RKL':
		Eq = q_samples - 1.
	elif measure == 'DV':
		Eq = log_sum_exp(q_samples, 0) - math.log(q_samples.size(0))
	elif measure == 'H2':
		Eq = torch.exp(q_samples) - 1.
	elif measure == 'W1':
		Eq = q_samples
	else:
		raise_measure_error(measure)

	if average:
		return Eq.mean()
	else:
		return Eq

def local_global_loss_(l_enc, g_enc, batch, measure):
	'''
	Args:
		l: Local feature map.
		g: Global features.
		measure: Type of f-divergence. For use with mode `fd`
		mode: Loss mode. Fenchel-dual `fd`, NCE `nce`, or Donsker-Vadadhan `dv`.
	Returns:
		torch.Tensor: Loss.
	'''
	num_graphs = g_enc.shape[0]
	num_nodes = l_enc.shape[0]

	pos_mask = torch.zeros((num_nodes, num_graphs)).cuda()
	neg_mask = torch.ones((num_nodes, num_graphs)).cuda()
	for nodeidx, graphidx in enumerate(batch):
		pos_mask[nodeidx][graphidx] = 1.
		neg_mask[nodeidx][graphidx] = 0.

	res = torch.mm(l_enc, g_enc.t())

	E_pos = get_positive_expectation(res * pos_mask, measure, average=False).sum()
	E_pos = E_pos / num_nodes
	E_neg = get_negative_expectation(res * neg_mask, measure, average=False).sum()
	E_neg = E_neg / (num_nodes * (num_graphs - 1))

	return E_neg - E_pos


class PriorDiscriminator(nn.Module):
	def __init__(self, input_dim):
		super().__init__()
		self.l0 = nn.Linear(input_dim, input_dim)
		self.l1 = nn.Linear(input_dim, input_dim)
		self.l2 = nn.Linear(input_dim, 1)

	def forward(self, x):
		h = F.relu(self.l0(x))
		h = F.relu(self.l1(h))
		return torch.sigmoid(self.l2(h))


class FF(nn.Module):
	def __init__(self, input_dim):
		super().__init__()
		self.block = nn.Sequential(
			nn.Linear(input_dim, input_dim),
			nn.ReLU(),
			nn.Linear(input_dim, input_dim),
			nn.ReLU(),
			nn.Linear(input_dim, input_dim),
			nn.ReLU()
		)
		self.linear_shortcut = nn.Linear(input_dim, input_dim)

	def forward(self, x):
		return self.block(x) + self.linear_shortcut(x)


class InfoGraph(nn.Module):
	def __init__(self, encoder, alpha=0.5, beta=1., gamma=.1, prior=True):
		super(InfoGraph, self).__init__()

		self.alpha = alpha
		self.beta = beta
		self.gamma = gamma
		self.prior = prior


		self.encoder = encoder
		self.embedding_dim = self.encoder.out_graph_dim

		self.local_d = FF(self.embedding_dim)
		self.global_d = FF(self.embedding_dim)

		if self.prior:
			self.prior_d = PriorDiscriminator(self.embedding_dim)

		self.init_emb()

	def init_emb(self):
		for m in self.modules():
			if isinstance(m, nn.Linear):
				torch.nn.init.xavier_uniform_(m.weight.data)
				if m.bias is not None:
					m.bias.data.fill_(0.0)

	def forward(self, batch, x, edge_index, edge_attr=None, edge_weight=None):
		# batch_size = data.num_graphs

		y, M = self.encoder(batch, x, edge_index, edge_attr, edge_weight)

		g_enc = self.global_d(y)
		l_enc = self.local_d(M)

		mode = 'fd'
		measure = 'JSD'
		local_global_loss = local_global_loss_(l_enc, g_enc, batch, measure)

		if self.prior:
			prior = torch.rand_like(y)
			term_a = torch.log(self.prior_d(prior)).mean()
			term_b = torch.log(1.0 - self.prior_d(y)).mean()
			PRIOR = - (term_a + term_b) * self.gamma
		else:
			PRIOR = 0

		return local_global_loss + PRIOR