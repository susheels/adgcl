from itertools import repeat

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset


def nx_to_graph_data_obj(g, center_id, allowable_features_downstream=None,
                         allowable_features_pretrain=None,
                         node_id_to_go_labels=None):
	"""
	Converts nx graph of PPI to pytorch geometric Data object.
	:param g: nx graph object of ego graph
	:param center_id: node id of center node in the ego graph
	:param allowable_features_downstream: list of possible go function node
	features for the downstream task. The resulting go_target_downstream node
	feature vector will be in this order.
	:param allowable_features_pretrain: list of possible go function node
	features for the pretraining task. The resulting go_target_pretrain node
	feature vector will be in this order.
	:param node_id_to_go_labels: dict that maps node id to a list of its
	corresponding go labels
	:return: pytorch geometric Data object with the following attributes:
	edge_attr
	edge_index
	x
	species_id
	center_node_idx
	go_target_downstream (only if node_id_to_go_labels is not None)
	go_target_pretrain (only if node_id_to_go_labels is not None)
	"""
	n_nodes = g.number_of_nodes()
	n_edges = g.number_of_edges()

	# nodes
	nx_node_ids = [n_i for n_i in g.nodes()]  # contains list of nx node ids
	# in a particular ordering. Will be used as a mapping to convert
	# between nx node ids and data obj node indices

	x = torch.tensor(np.ones(n_nodes).reshape(-1, 1), dtype=torch.float)
	# we don't have any node labels, so set to dummy 1. dim n_nodes x 1

	center_node_idx = nx_node_ids.index(center_id)
	center_node_idx = torch.tensor([center_node_idx], dtype=torch.long)

	# edges
	edges_list = []
	edge_features_list = []
	for node_1, node_2, attr_dict in g.edges(data=True):
		edge_feature = [attr_dict['w1'], attr_dict['w2'], attr_dict['w3'],
		                attr_dict['w4'], attr_dict['w5'], attr_dict['w6'],
		                attr_dict['w7'], 0, 0]  # last 2 indicate self-loop
		# and masking
		edge_feature = np.array(edge_feature, dtype=int)
		# convert nx node ids to data obj node index
		i = nx_node_ids.index(node_1)
		j = nx_node_ids.index(node_2)
		edges_list.append((i, j))
		edge_features_list.append(edge_feature)
		edges_list.append((j, i))
		edge_features_list.append(edge_feature)

	# data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
	edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

	# data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
	edge_attr = torch.tensor(np.array(edge_features_list),
	                         dtype=torch.float)

	try:
		species_id = int(nx_node_ids[0].split('.')[0])  # nx node id is of the form:
		# species_id.protein_id
		species_id = torch.tensor([species_id], dtype=torch.long)
	except:  # occurs when nx node id has no species id info. For the extract
		# substructure context pair transform, where we convert a data obj to
		# a nx graph obj (which does not have original node id info)
		species_id = torch.tensor([0], dtype=torch.long)  # dummy species
	# id is 0

	# construct data obj
	data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
	data.species_id = species_id
	data.center_node_idx = center_node_idx

	if node_id_to_go_labels:  # supervised case with go node labels
		# Construct a dim n_pretrain_go_classes tensor and a
		# n_downstream_go_classes tensor for the center node. 0 is no data
		# or negative, 1 is positive.
		downstream_go_node_feature = [0] * len(allowable_features_downstream)
		pretrain_go_node_feature = [0] * len(allowable_features_pretrain)
		if center_id in node_id_to_go_labels:
			go_labels = node_id_to_go_labels[center_id]
			# get indices of allowable_features_downstream that match with elements
			# in go_labels
			_, node_feature_indices, _ = np.intersect1d(
				allowable_features_downstream, go_labels, return_indices=True)
			for idx in node_feature_indices:
				downstream_go_node_feature[idx] = 1
			# get indices of allowable_features_pretrain that match with
			# elements in go_labels
			_, node_feature_indices, _ = np.intersect1d(
				allowable_features_pretrain, go_labels, return_indices=True)
			for idx in node_feature_indices:
				pretrain_go_node_feature[idx] = 1
		data.go_target_downstream = torch.tensor(np.array(downstream_go_node_feature),
		                                         dtype=torch.long)
		data.go_target_pretrain = torch.tensor(np.array(pretrain_go_node_feature),
		                                       dtype=torch.long)

	return data


def graph_data_obj_to_nx(data):
	"""
	Converts pytorch geometric Data obj to network x data object.
	:param data: pytorch geometric Data object
	:return: nx graph object
	"""
	G = nx.Graph()

	# edges
	edge_index = data.edge_index.cpu().numpy()
	edge_attr = data.edge_attr.cpu().numpy()
	n_edges = edge_index.shape[1]
	for j in range(0, n_edges, 2):
		begin_idx = int(edge_index[0, j])
		end_idx = int(edge_index[1, j])
		w1, w2, w3, w4, w5, w6, w7, _, _ = edge_attr[j].astype(bool)
		if not G.has_edge(begin_idx, end_idx):
			G.add_edge(begin_idx, end_idx, w1=w1, w2=w2, w3=w3, w4=w4, w5=w5,
			           w6=w6, w7=w7)

	# # add center node id information in final nx graph object
	# nx.set_node_attributes(G, {data.center_node_idx.item(): True}, 'is_centre')

	return G


class BioDataset_aug(InMemoryDataset):
	def __init__(self,
	             root,
	             data_type,
	             empty=False,
	             transform=None,
	             pre_transform=None,
	             pre_filter=None,
	             aug="none", aug_ratio=None):
		"""
		Adapted from qm9.py. Disabled the download functionality
		:param root: the data directory that contains a raw and processed dir
		:param data_type: either supervised or unsupervised
		:param empty: if True, then will not load any data obj. For
		initializing empty dataset
		:param transform:
		:param pre_transform:
		:param pre_filter:
		"""
		self.root = root
		self.data_type = data_type
		self.aug = aug
		self.aug_ratio = aug_ratio

		super(BioDataset_aug, self).__init__(root, transform, pre_transform, pre_filter)
		if not empty:
			self.data, self.slices = torch.load(self.processed_paths[0])

	def get(self, idx):
		data = Data()
		for key in self.data.keys:
			item, slices = self.data[key], self.slices[key]
			s = list(repeat(slice(None), item.dim()))
			s[data.cat_dim(key, item)] = slice(slices[idx],
			                                   slices[idx + 1])
			data[key] = item[s]

		if self.aug == 'dropN':
			data = drop_nodes(data, self.aug_ratio)
		elif self.aug == 'permE':
			data = permute_edges(data, self.aug_ratio)
		elif self.aug == 'maskN':
			data = mask_nodes(data, self.aug_ratio)
		elif self.aug == 'subgraph':
			data = subgraph(data, self.aug_ratio)
		elif self.aug == 'random':
			n = np.random.randint(3)
			if n == 0:
				data = drop_nodes(data, self.aug_ratio)
			elif n == 1:
				data = subgraph(data, self.aug_ratio)
			elif n == 2:
				data = permute_edges(data, self.aug_ratio)
			else:
				print('augmentation error')
				assert False
		elif self.aug == 'none':
			None
		else:
			print('augmentation error')
			assert False

		return data

	@property
	def raw_file_names(self):
		# raise NotImplementedError('Data is assumed to be processed')
		if self.data_type == 'supervised':  # 8 labelled species
			file_name_list = ['3702', '6239', '511145', '7227', '9606', '10090', '4932', '7955']
		else:  # unsupervised: 8 labelled species, and 42 top unlabelled species by n_nodes.
			file_name_list = ['3702', '6239', '511145', '7227', '9606', '10090',
			                  '4932', '7955', '3694', '39947', '10116', '443255', '9913', '13616',
			                  '3847', '4577', '8364', '9823', '9615', '9544', '9796', '3055', '7159',
			                  '9031', '7739', '395019', '88036', '9685', '9258', '9598', '485913',
			                  '44689', '9593', '7897', '31033', '749414', '59729', '536227', '4081',
			                  '8090', '9601', '749927', '13735', '448385', '457427', '3711', '479433',
			                  '479432', '28377', '9646']
		return file_name_list

	@property
	def processed_file_names(self):
		return 'geometric_data_processed.pt'

	def download(self):
		raise NotImplementedError('Must indicate valid location of raw data. '
		                          'No download allowed')

	def process(self):
		raise NotImplementedError('Data is assumed to be processed')


def drop_nodes(data, aug_ratio):
	node_num, _ = data.x.size()
	_, edge_num = data.edge_index.size()
	drop_num = int(node_num * aug_ratio)

	idx_perm = np.random.permutation(node_num)

	idx_drop = idx_perm[:drop_num]
	idx_nondrop = idx_perm[drop_num:]
	idx_nondrop.sort()
	idx_dict = {idx_nondrop[n]: n for n in list(range(idx_nondrop.shape[0]))}

	edge_index = data.edge_index.numpy()
	edge_mask = np.array(
		[n for n in range(edge_num) if not (edge_index[0, n] in idx_drop or edge_index[1, n] in idx_drop)])

	edge_index = [[idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]] for n in range(edge_num) if
	              (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)]
	try:
		data.edge_index = torch.tensor(edge_index).transpose_(0, 1)
		data.x = data.x[idx_nondrop]
		data.edge_attr = data.edge_attr[edge_mask]
	except:
		data = data

	return data


def permute_edges(data, aug_ratio):
	node_num, _ = data.x.size()
	_, edge_num = data.edge_index.size()
	permute_num = int(edge_num * aug_ratio)
	edge_index = data.edge_index.numpy()

	idx_delete = np.random.choice(edge_num, (edge_num - permute_num), replace=False)
	data.edge_index = data.edge_index[:, idx_delete]
	data.edge_attr = data.edge_attr[idx_delete]

	return data


def mask_nodes(data, aug_ratio):
	node_num, feat_dim = data.x.size()
	mask_num = int(node_num * aug_ratio)

	token = data.x.mean(dim=0)
	idx_mask = np.random.choice(node_num, mask_num, replace=False)
	data.x[idx_mask] = torch.tensor(token, dtype=torch.float32)

	return data


def subgraph(data, aug_ratio):
	node_num, _ = data.x.size()
	_, edge_num = data.edge_index.size()
	sub_num = int(node_num * aug_ratio)

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
	edge_mask = np.array(
		[n for n in range(edge_num) if (edge_index[0, n] in idx_nondrop and edge_index[1, n] in idx_nondrop)])

	edge_index = data.edge_index.numpy()
	edge_index = [[idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]] for n in range(edge_num) if
	              (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)]
	try:
		data.edge_index = torch.tensor(edge_index).transpose_(0, 1)
		data.x = data.x[idx_nondrop]
		data.edge_attr = data.edge_attr[edge_mask]
	except:
		data = data

	return data


class BioDataset(InMemoryDataset):
	def __init__(self,
	             root,
	             data_type,
	             empty=False,
	             transform=None,
	             pre_transform=None,
	             pre_filter=None):
		"""
		Adapted from qm9.py. Disabled the download functionality
		:param root: the data directory that contains a raw and processed dir
		:param data_type: either supervised or unsupervised
		:param empty: if True, then will not load any data obj. For
		initializing empty dataset
		:param transform:
		:param pre_transform:
		:param pre_filter:
		"""
		self.root = root
		self.data_type = data_type

		super(BioDataset, self).__init__(root, transform, pre_transform, pre_filter)
		if not empty:
			self.data, self.slices = torch.load(self.processed_paths[0])

	@property
	def raw_file_names(self):
		# raise NotImplementedError('Data is assumed to be processed')
		if self.data_type == 'supervised':  # 8 labelled species
			file_name_list = ['3702', '6239', '511145', '7227', '9606', '10090', '4932', '7955']
		else:  # unsupervised: 8 labelled species, and 42 top unlabelled species by n_nodes.
			file_name_list = ['3702', '6239', '511145', '7227', '9606', '10090',
			                  '4932', '7955', '3694', '39947', '10116', '443255', '9913', '13616',
			                  '3847', '4577', '8364', '9823', '9615', '9544', '9796', '3055', '7159',
			                  '9031', '7739', '395019', '88036', '9685', '9258', '9598', '485913',
			                  '44689', '9593', '7897', '31033', '749414', '59729', '536227', '4081',
			                  '8090', '9601', '749927', '13735', '448385', '457427', '3711', '479433',
			                  '479432', '28377', '9646']
		return file_name_list

	@property
	def processed_file_names(self):
		return 'geometric_data_processed.pt'

	def download(self):
		raise NotImplementedError('Must indicate valid location of raw data. '
		                          'No download allowed')

	def process(self):
		raise NotImplementedError('Data is assumed to be processed')


if __name__ == "__main__":
	root_supervised = '../original_datasets/transfer/bio/supervised'

	d_supervised = BioDataset(root_supervised, data_type='supervised')

	print(d_supervised)

	root_unsupervised = '../original_datasets/transfer/bio/unsupervised'
	d_unsupervised = BioDataset(root_unsupervised, data_type='unsupervised')

	print(d_unsupervised)

