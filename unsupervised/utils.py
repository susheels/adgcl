import copy

import numpy as np
import torch
from torch_geometric.data import Data


def initialize_edge_weight(data):
	data.edge_weight = torch.ones(data.edge_index.shape[1], dtype=torch.float)
	return data

def initialize_node_features(data):
	num_nodes = int(data.edge_index.max()) + 1
	data.x = torch.ones((num_nodes, 1))
	return data

def set_tu_dataset_y_shape(data):
	num_tasks = 1
	data.y = data.y.unsqueeze(num_tasks)
	return data

def drop_nodes(data, ratio=0.1):
	if hasattr(data, "x"):
		node_num, _ = data.x.size()
	else:
		node_num = int(data.edge_index.max()) + 1

	_, edge_num = data.edge_index.size()
	drop_num = int(node_num * ratio)
	idx_drop = np.random.choice(node_num, drop_num, replace=False)
	idx_nondrop = [n for n in range(node_num) if not n in idx_drop]
	idx_dict = {idx_nondrop[n]: n for n in list(range(node_num - drop_num))}

	x_aug = data.x[idx_nondrop]

	remove_edge_mask = torch.zeros(edge_num, dtype=torch.bool)

	for n_idx in idx_drop:
		temp_mask = torch.logical_or(data.edge_index[0] == n_idx, data.edge_index[1] == n_idx)
		remove_edge_mask = torch.logical_or(temp_mask, remove_edge_mask)

	edge_index_aug = data.edge_index[:, ~remove_edge_mask]

	edge_num_aug = edge_index_aug.shape[1]
	edge_index_aug = edge_index_aug.numpy()
	edge_index_aug = [[idx_dict[edge_index_aug[0, n]], idx_dict[edge_index_aug[1, n]]] for n in range(edge_num_aug) if
	                  not edge_index_aug[0, n] == edge_index_aug[1, n]]
	try:
		edge_index_aug = torch.tensor(edge_index_aug).transpose_(0, 1)
	except:
		#         print(data)
		#         print(data.edge_index)
		#         print(idx_drop)
		#         print(idx_nondrop)
		#         print(idx_dict)
		#         print(~remove_edge_mask)
		#         print(edge_index_aug)
		#         print("Removal of all edges happened ... Returning original data")
		edge_index_aug = torch.tensor([[], []], dtype=torch.long)
	#         data_aug = Data(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr)
	#         return data_aug

	if hasattr(data, "edge_attr")  and data.edge_attr is not None:
		edge_attr_aug = data.edge_attr[~remove_edge_mask]
		data_aug = Data(x=x_aug, edge_index=edge_index_aug, edge_attr=edge_attr_aug)
	else:
		data_aug = Data(x=x_aug, edge_index=edge_index_aug)

	return data_aug


def subgraph(data, ratio=0.2):
	node_num, _ = data.x.size()
	_, edge_num = data.edge_index.size()
	sub_num = int(node_num * ratio)

	edge_index = data.edge_index.numpy()

	idx_sub = [np.random.randint(node_num, size=1)[0]]
	idx_neigh = set([n for n in edge_index[1][edge_index[0] == idx_sub[0]]])

	count = 0
	while len(idx_sub) <= sub_num:
		count = count + 1
		if count > node_num:
			break
		if len(idx_neigh) == 0:
			break
		sample_node = np.random.choice(list(idx_neigh))
		if sample_node in idx_sub:
			continue
		idx_sub.append(sample_node)
		idx_neigh.union(set([n for n in edge_index[1][edge_index[0] == idx_sub[-1]]]))

	idx_drop = [n for n in range(node_num) if not n in idx_sub]
	idx_nondrop = idx_sub
	idx_dict = {idx_nondrop[n]: n for n in list(range(len(idx_nondrop)))}

	x_aug = data.x[idx_nondrop]

	remove_edge_mask = torch.zeros(edge_num, dtype=torch.bool)

	for n_idx in idx_drop:
		temp_mask = torch.logical_or(data.edge_index[0] == n_idx, data.edge_index[1] == n_idx)
		remove_edge_mask = torch.logical_or(temp_mask, remove_edge_mask)

	edge_index_aug = data.edge_index[:, ~remove_edge_mask]
	edge_num_aug = edge_index_aug.shape[1]
	edge_index_aug = edge_index_aug.numpy()
	edge_index_aug = [[idx_dict[edge_index_aug[0, n]], idx_dict[edge_index_aug[1, n]]] for n in range(edge_num_aug) if
	                  not edge_index_aug[0, n] == edge_index_aug[1, n]]
	try:
		edge_index_aug = torch.tensor(edge_index_aug).transpose_(0, 1)
	except:
		#         print(data)
		#         print(data.edge_index)
		#         print(idx_drop)
		#         print(idx_nondrop)
		#         print(idx_dict)
		#         print(~remove_edge_mask)
		#         print(edge_index_aug)
		#         print("Removal of all edges happened ... Returning original data")
		edge_index_aug = torch.tensor([[], []], dtype=torch.long)

	if hasattr(data, "edge_attr")  and data.edge_attr is not None:
		edge_attr_aug = data.edge_attr[~remove_edge_mask]
		data_aug = Data(x=x_aug, edge_index=edge_index_aug, edge_attr=edge_attr_aug)
	else:
		data_aug = Data(x=x_aug, edge_index=edge_index_aug)

	# edge_index = [[idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]] for n in range(edge_num) if (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)]
	# edge_index = [[edge_index[0, n], edge_index[1, n]] for n in range(edge_num) if (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)] + [[n, n] for n in idx_nondrop]
	# data.edge_index = torch.tensor(edge_index).transpose_(0, 1)

	return data_aug


def perturb_edges(data, edge_ratio=0.1):
	node_num, _ = data.x.size()
	_, edge_num = data.edge_index.size()
	permute_num = int(edge_num * edge_ratio)

	edge_index = data.edge_index.transpose(0, 1).numpy()
	mask = np.random.choice(edge_num, edge_num - permute_num, replace=False)
	edge_index_aug = edge_index[mask]
	edge_index_aug = torch.tensor(edge_index_aug).transpose_(0, 1)

	if hasattr(data, "edge_attr") and data.edge_attr is not None:
		edge_attr_aug = data.edge_attr[mask]
		data_aug = Data(x=data.x, edge_index=edge_index_aug, edge_attr=edge_attr_aug)
	else:
		data_aug = Data(x=data.x, edge_index=edge_index_aug)

	return data_aug

def get_aug_pair(data):
	n = np.random.randint(2)
	if n == 0:
		data_aug = drop_nodes(data)
	elif n == 1:
		data_aug = subgraph(data)
	else:
		raise NotImplementedError
	return data, data_aug


def biased_perturb_edges(data, edge_ratio=0.1):
	node_num, _ = data.x.size()
	_, edge_num = data.edge_index.size()

	permute_num = int(edge_num * edge_ratio)
	core_edges_mask = data.core.numpy().astype(bool)

	edge_index = data.edge_index.transpose(0, 1).numpy()

	mask = np.random.choice(edge_num, edge_num - permute_num, replace=False)
	mask = np.union1d(mask, np.arange(edge_num)[core_edges_mask])
	edge_index_aug = edge_index[mask]
	edge_index_aug = torch.tensor(edge_index_aug).transpose_(0, 1)

	if hasattr(data, "edge_attr") and data.edge_attr is not None:
		edge_attr_aug = data.edge_attr[mask]
		data_aug = Data(x=data.x, edge_index=edge_index_aug, edge_attr=edge_attr_aug)
	else:
		data_aug = Data(x=data.x, edge_index=edge_index_aug)

	return data_aug

class BiasedAugTransformer():
	def __init__(self, type_code="pedges", aug_ratio=0.1):
		self.tc = type_code
		self.aug_ratio = aug_ratio
		# self.dn_ratio = aug_ratio
		# self.sub_ratio = aug_ratio
		# self.edge_ratio = aug_r atio
		print("Biased Augmentation Type: %s, Aug Ratio: %f, Using Edge Mask Name core " % (self.tc, self.aug_ratio))

	def __call__(self, data):
		if self.tc == "copy":
			data_aug = copy.deepcopy(data)
		elif self.tc == "pedges":
			data_aug = biased_perturb_edges(data, self.aug_ratio)
		else:
			raise NotImplementedError

		return data, data_aug


class MyAugTransformer(object):
	def __init__(self, type_code="dnodes", aug_ratio=0.1):
		self.tc = type_code
		self.aug_ratio = aug_ratio
		# self.dn_ratio = aug_ratio
		# self.sub_ratio = aug_ratio
		# self.edge_ratio = aug_ratio
		print("Augmentation Type: %s, Aug Ratio: %f " %(self.tc, self.aug_ratio))

	def __call__(self, data):
		if self.tc == "copy":
			data_aug = copy.deepcopy(data)
		elif self.tc == "dnodes":
			data_aug = drop_nodes(data, self.aug_ratio)
		elif self.tc == "pedges":
			data_aug = perturb_edges(data, self.aug_ratio)
		elif self.tc == "subgraph":
			data_aug = subgraph(data, self.aug_ratio)
		elif self.tc == "dnodes+subgraph":
			n = np.random.randint(2)
			if n == 0:
				data_aug = drop_nodes(data, self.aug_ratio)
			elif n == 1:
				data_aug = subgraph(data, self.aug_ratio)
			else:
				raise NotImplementedError
		else:
			raise NotImplementedError

		return data, data_aug