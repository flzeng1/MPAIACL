import random

import numpy as np
import torch
from torch_geometric.loader import DataLoader


def get_dataloader(dataset, args):
    train_loader = DataLoader(dataset["train"], batch_size=args.batch_size, shuffle=True, num_workers=0,
                              worker_init_fn=_init_fn)
    valid_loader_ood = DataLoader(dataset["val"], batch_size=args.batch_size, shuffle=False, num_workers=0,
                                  worker_init_fn=_init_fn)
    test_loader_ood = DataLoader(dataset["test"], batch_size=args.batch_size, shuffle=False, num_workers=0,
                                 worker_init_fn=_init_fn)

    return train_loader, valid_loader_ood, test_loader_ood



def _init_fn(worker_id):
    random.seed(10 + worker_id)
    np.random.seed(10 + worker_id)
    torch.manual_seed(10 + worker_id)
    torch.cuda.manual_seed(10 + worker_id)
    torch.cuda.manual_seed_all(10 + worker_id)


def get_h_causaler(model, x_encode, data):
    x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
    causaler_output = model.causaler(data)
    node_cau, edge_cau = causaler_output["node_key"], causaler_output["edge_key"]
    h_node_cau = model.graph_backs(x_encode, edge_index, edge_attr, batch, node_cau, edge_cau)
    h_graph_cau = model.pool(h_node_cau, batch)

    # get reg
    node_cau_num, node_env_num = causaler_output["node_key_num"], causaler_output["node_env_num"]
    edge_cau_num, edge_env_num = causaler_output["edge_key_num"], causaler_output["edge_env_num"]
    cau_node_reg = model.reg_mask_loss(node_cau_num, node_env_num, model.cau_gamma, model.causaler.non_zero_node_ratio)
    cau_edge_reg = model.reg_mask_loss(edge_cau_num, edge_env_num, model.cau_gamma, model.causaler.non_zero_edge_ratio)
    cau_loss_reg = cau_node_reg + cau_edge_reg

    return node_cau, edge_cau, h_graph_cau, cau_loss_reg


def get_h_attack(model, x_encode, data):
    x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
    attacker_output = model.attacker(data)
    node_adv, edge_adv = attacker_output["node_key"], attacker_output["edge_key"]
    h_node_adv = model.graph_backs(x_encode, edge_index, edge_attr, batch, node_adv, edge_adv)
    h_graph_adv = model.pool(h_node_adv, batch)

    return node_adv, edge_adv, h_graph_adv


def get_h_DA(model, x_encode, data, node_adv, edge_adv, node_cau, edge_cau):
    x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
    node_com = (1 - node_cau) * node_adv + node_cau
    edge_com = (1 - edge_cau) * edge_adv + edge_cau
    h_node_com = model.graph_backs(x_encode, edge_index, edge_attr, batch, node_com, edge_com)
    h_graph_com = model.pool(h_node_com, batch)

    return h_graph_com

def eval(model, evaluator, loader, device):
    model.eval()
    y_true = []
    y_pred = []
    for step, batch in enumerate(loader):
        batch = batch.to(device)
        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model.forward_causal(batch)
            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())
    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()
    input_dict = {"y_true": y_true, "y_pred": y_pred}
    output = evaluator.eval(input_dict)
    return output