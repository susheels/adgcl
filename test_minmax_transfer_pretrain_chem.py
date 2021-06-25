import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import DataLoader
from torch_geometric.transforms import Compose
from torch_scatter import scatter
from tqdm import tqdm

from datasets import MoleculeDataset
from transfer.learning import GInfoMinMax, ViewLearner
from transfer.model import GNN
from unsupervised.utils import initialize_edge_weight


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def train(args, model, view_learner, device, dataset, model_optimizer, view_optimizer):
    dataset = dataset.shuffle()

    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=8, shuffle=False)

    model.train()

    model_loss_all = 0
    view_loss_all = 0
    reg_all = 0

    for step, batch in enumerate(dataloader):
        batch = batch.to(device)

        # train view to maximize contrastive loss
        view_learner.train()
        view_learner.zero_grad()
        model.eval()

        x = model(batch.batch, batch.x, batch.edge_index, batch.edge_attr, None)

        edge_logits = view_learner(batch.batch, batch.x, batch.edge_index, batch.edge_attr)

        temperature = 1.0
        bias = 0.0 + 0.0001  # If bias is 0, we run into problems
        eps = (bias - (1 - bias)) * torch.rand(edge_logits.size()) + (1 - bias)
        gate_inputs = torch.log(eps) - torch.log(1 - eps)
        gate_inputs = gate_inputs.to(device)
        gate_inputs = (gate_inputs + edge_logits) / temperature
        batch_aug_edge_weight = torch.sigmoid(gate_inputs).squeeze()

        x_aug = model(batch.batch, batch.x, batch.edge_index, batch.edge_attr, batch_aug_edge_weight)

        # regularization

        row, col = batch.edge_index
        edge_batch = batch.batch[row]
        edge_drop_out_prob = 1 - batch_aug_edge_weight

        uni, edge_batch_num = edge_batch.unique(return_counts=True)
        sum_pe = scatter(edge_drop_out_prob, edge_batch, reduce="sum")

        reg = []
        for b_id in range(args.batch_size):
            if b_id in uni:
                num_edges = edge_batch_num[uni.tolist().index(b_id)]
                reg.append(sum_pe[b_id] / num_edges)
            else:
                # means no edges in that graph. So don't include.
                pass
        num_graph_with_edges = len(reg)
        reg = torch.stack(reg)
        reg = reg.mean()

        view_loss = model.calc_loss(x, x_aug, temperature=0.2) - (args.reg_lambda * reg)
        view_loss_all += view_loss.item() * batch.num_graphs
        reg_all += reg.item()
        # gradient ascent formulation
        (-view_loss).backward()
        view_optimizer.step()

        model.train()
        view_learner.eval()
        # train (model) to minimize contrastive loss
        model.zero_grad()

        x = model(batch.batch, batch.x, batch.edge_index, batch.edge_attr, None)
        edge_logits = view_learner(batch.batch, batch.x, batch.edge_index, batch.edge_attr)


        temperature = 1.0
        bias = 0.0 + 0.0001  # If bias is 0, we run into problems
        eps = (bias - (1 - bias)) * torch.rand(edge_logits.size()) + (1 - bias)
        gate_inputs = torch.log(eps) - torch.log(1 - eps)
        gate_inputs = gate_inputs.to(device)
        gate_inputs = (gate_inputs + edge_logits) / temperature
        batch_aug_edge_weight = torch.sigmoid(gate_inputs).squeeze().detach()

        x_aug = model(batch.batch, batch.x, batch.edge_index, batch.edge_attr, batch_aug_edge_weight)

        model_loss = model.calc_loss(x, x_aug, temperature=0.2)
        model_loss_all += model_loss.item() * batch.num_graphs
        # standard gradient descent formulation
        model_loss.backward()
        model_optimizer.step()

    fin_model_loss = model_loss_all / len(dataloader)
    fin_view_loss = view_loss_all / len(dataloader)
    fin_reg = reg_all / len(dataloader)

    return fin_model_loss, fin_view_loss, fin_reg

def run(args):
    Path("./models_minmax/chem").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info("Using Device: %s" % device)
    logging.info("Seed: %d" % args.seed)
    logging.info(args)
    setup_seed(args.seed)

    my_transforms = Compose([initialize_edge_weight])
    dataset = MoleculeDataset("original_datasets/transfer/"+args.dataset, dataset=args.dataset,
                              transform=my_transforms)
    model = GInfoMinMax(
        GNN(num_layer=args.num_gc_layers, emb_dim=args.emb_dim, JK="last", drop_ratio=args.drop_ratio, gnn_type="gin"),
        proj_hidden_dim=args.emb_dim).to(device)
    model_optimizer = torch.optim.Adam(model.parameters(), lr=args.model_lr)

    view_learner = ViewLearner(
        GNN(num_layer=args.num_gc_layers, emb_dim=args.emb_dim, JK="last", drop_ratio=args.drop_ratio, gnn_type="gin"),
        mlp_edge_model_dim=args.mlp_edge_model_dim).to(device)

    view_optimizer = torch.optim.Adam(view_learner.parameters(), lr=args.view_lr)

    for epoch in tqdm(range(1, args.epochs)):
        logging.info('====epoch {}'.format(epoch))

        model_loss, view_loss, reg = train(args, model, view_learner, device, dataset, model_optimizer, view_optimizer)
        logging.info(
            'Epoch {}, Model Loss {}, View Loss {}, Reg {}'.format(epoch, model_loss, view_loss, reg))

        if epoch % 1 == 0:
            torch.save(model.gnn.state_dict(), "./models_minmax/chem/pretrain_minmax_seed_"+str(args.seed)+"_reg_"+str(args.reg_lambda)+"_epoch_"+ str(epoch)+".pth")





def arg_parse():
    parser = argparse.ArgumentParser(description='Transfer Learning AD-GCL Pretrain on ZINC 2M')

    parser.add_argument('--dataset', type=str, default='zinc_standard_agent',
                        help='Dataset')
    parser.add_argument('--model_lr', type=float, default=0.001,
                        help='Model Learning rate.')
    parser.add_argument('--view_lr', type=float, default=0.001,
                        help='View Learning rate.')
    parser.add_argument('--num_gc_layers', type=int, default=5,
                        help='Number of GNN layers before pooling')
    parser.add_argument('--pooling_type', type=str, default='standard',
                        help='GNN Pooling Type Standard/Layerwise')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimension')
    parser.add_argument('--mlp_edge_model_dim', type=int, default=64,
                        help='embedding dimension')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch size')
    parser.add_argument('--drop_ratio', type=float, default=0.0,
                        help='Dropout Ratio / Probability')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Train Epochs')
    parser.add_argument('--reg_lambda', type=float, default=5.0, help='View Learner Edge Perturb Regularization Strength')

    parser.add_argument('--seed', type=int, default=0)

    return parser.parse_args()


if __name__ == '__main__':

    args = arg_parse()
    run(args)