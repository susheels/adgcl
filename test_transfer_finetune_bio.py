import argparse
import logging
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score

from datasets import BioDataset
from transfer import BioGNN_graphpred
from transfer.utils import DataLoaderFinetune
from transfer.utils import bio_random_split, bio_species_split


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def train( model, device, loader, optimizer, criterion):
    model.train()
    loss_all = 0
    for step, batch in enumerate(loader):
        batch = batch.to(device)
        pred = model(batch)
        y = batch.go_target_downstream.view(pred.shape).to(torch.float64)

        optimizer.zero_grad()
        loss = criterion(pred.double(), y)
        loss.backward()

        optimizer.step()
        loss_all += loss.item() * batch.num_graphs

    return loss_all / len(loader)

def eval(model, device, loader):
    model.eval()
    y_true = []
    y_scores = []

    for step, batch in enumerate(loader):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)

        y_true.append(batch.go_target_downstream.view(pred.shape).detach().cpu())
        y_scores.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_scores = torch.cat(y_scores, dim=0).numpy()

    roc_list = []
    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            roc_list.append(roc_auc_score(y_true[:, i], y_scores[:, i]))
        else:
            roc_list.append(np.nan)

    return np.array(roc_list).mean()  # y_true.shape[1]

def arg_parse():
    parser = argparse.ArgumentParser(description='Finetuning Bio after pre-training of graph neural networks')
    parser.add_argument('--dataset', type=str, default="supervised")
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--input_model_file', type=str, default='', help='filename to read the model (if there is any)')
    parser.add_argument('--filename', type=str, default='', help='output filename')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--seed', type=int, default=0, help="Seed for running experiments")
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for dataset loading')
    parser.add_argument('--eval_train', type=int, default=1, help='evaluating training or not')
    parser.add_argument('--split', type=str, default="species", help='Random or species split')
    return parser.parse_args()

def run(args):
    # Training settings
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info("Using Device: %s" % device)
    logging.info("Seed: %d" % args.seed)
    logging.info(args)

    setup_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    root_supervised = 'original_datasets/transfer/bio/supervised'

    dataset = BioDataset(root_supervised, data_type='supervised')

    logging.info(dataset)

    node_num = 0
    edge_num = 0
    for d in dataset:
        node_num += d.x.size()[0]
        edge_num += d.edge_index.size()[1]
    logging.info(node_num / len(dataset))
    logging.info(edge_num / len(dataset))

    if args.split == "random":
        logging.info("random splitting")
        train_dataset, valid_dataset, test_dataset = bio_random_split(dataset, seed=0)
    elif args.split == "species":
        trainval_dataset, test_dataset = bio_species_split(dataset)
        train_dataset, valid_dataset, _ = bio_random_split(trainval_dataset, seed=0, frac_train=0.85,
                                                       frac_valid=0.15, frac_test=0)
        test_dataset_broad, test_dataset_none, _ = bio_random_split(test_dataset, seed=0, frac_train=0.5,
                                                                frac_valid=0.5, frac_test=0)
        logging.info("species splitting")
    else:
        raise ValueError("Unknown split name.")

    train_loader = DataLoaderFinetune(train_dataset, batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.num_workers)
    val_loader = DataLoaderFinetune(valid_dataset, batch_size=10 * args.batch_size, shuffle=False,
                                    num_workers=args.num_workers)

    if args.split == "random":
        test_loader = DataLoaderFinetune(test_dataset, batch_size=10 * args.batch_size, shuffle=False,
                                         num_workers=args.num_workers)
    else:
        ### for species splitting
        test_easy_loader = DataLoaderFinetune(test_dataset_broad, batch_size=10 * args.batch_size, shuffle=False,
                                              num_workers=args.num_workers)
        test_hard_loader = DataLoaderFinetune(test_dataset_none, batch_size=10 * args.batch_size, shuffle=False,
                                              num_workers=args.num_workers)

    num_tasks = len(dataset[0].go_target_downstream)

    logging.info(train_dataset[0])

    # set up model
    model = BioGNN_graphpred(args.num_layer, args.emb_dim, num_tasks, JK=args.JK, drop_ratio=args.dropout_ratio,
                          graph_pooling=args.graph_pooling, gnn_type=args.gnn_type)

    if not args.input_model_file == "":
        model.from_pretrained(args.input_model_file)

    model.to(device)

    # set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)

    criterion = nn.BCEWithLogitsLoss()

    train_acc_list = []
    val_acc_list = []

    ### for random splitting
    test_acc_list = []

    ### for species splitting
    test_acc_easy_list = []
    test_acc_hard_list = []

    for epoch in range(1, args.epochs + 1):

        loss = train(model, device, train_loader, optimizer, criterion)
        logging.info("====epoch {} SupervisedLoss {}".format(epoch, loss))

        logging.info("====Evaluation")
        if args.eval_train:
            train_acc = eval(model, device, train_loader)
        else:
            train_acc = 0
            # print("ommitting training evaluation")
        val_acc = eval(model, device, val_loader)

        val_acc_list.append(val_acc)
        train_acc_list.append(train_acc)

        if args.split == "random":
            test_acc = eval(model, device, test_loader)
            test_acc_list.append(test_acc)
            logging.info("EvalTrain: {} EvalVal: {} EvalTest: {}".format(train_acc, val_acc, test_acc))
        else:
            test_acc_easy = eval(model, device, test_easy_loader)
            test_acc_hard = eval(model, device, test_hard_loader)
            test_acc_easy_list.append(test_acc_easy)
            test_acc_hard_list.append(test_acc_hard)
            logging.info("EvalTrain: {} EvalVal: {} EvalTestEasy: {} EvalTestHard: {} ".format(train_acc, val_acc, test_acc_easy, test_acc_hard))

    best_val_epoch = np.argmax(np.array(val_acc_list))
    best_train = max(train_acc_list)
    logging.info('FinishedTraining!')
    logging.info('BestEpoch: {}'.format(best_val_epoch))
    logging.info('BestTrainScore: {}'.format(best_train))
    logging.info('BestValidationScore: {}'.format(val_acc_list[best_val_epoch]))
    if args.split == "random":
        logging.info('FinalTestScore: {}'.format(test_acc_list[best_val_epoch]))
        return val_acc_list[best_val_epoch], test_acc_list[best_val_epoch]
    else:
        logging.info('FinalTestEasyScore: {}'.format(test_acc_easy_list[best_val_epoch]))
        logging.info('FinalTestHardScore: {}'.format(test_acc_hard_list[best_val_epoch]))

        return val_acc_list[best_val_epoch], test_acc_hard_list[best_val_epoch]


if __name__ == "__main__":
    args = arg_parse()
    run(args)