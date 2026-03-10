import torch
from torch_geometric.loader import DataLoader


def get_h_causaler(model, x_encode, data):
    x, edge_index, batch = data.x, data.edge_index, data.batch

    causaler_output = model.causaler(data)
    node_cau, edge_cau = causaler_output["node_key"], causaler_output["edge_key"]
    h_node_cau = model.graph_backs(x_encode, edge_index, node_cau, edge_cau)
    h_graph_cau = model.pool(h_node_cau, batch)

    # get reg
    node_cau_num, node_env_num = causaler_output["node_key_num"], causaler_output["node_env_num"]
    edge_cau_num, edge_env_num = causaler_output["edge_key_num"], causaler_output["edge_env_num"]
    cau_node_reg = model.reg_mask_loss(node_cau_num, node_env_num, model.cau_gamma, model.causaler.non_zero_node_ratio)
    cau_edge_reg = model.reg_mask_loss(edge_cau_num, edge_env_num, model.cau_gamma, model.causaler.non_zero_edge_ratio)
    cau_loss_reg = cau_node_reg + cau_edge_reg

    return node_cau, edge_cau, h_graph_cau, cau_loss_reg

def get_h_attack(model, x_encode, data):
    x, edge_index, batch = data.x, data.edge_index, data.batch

    attacker_output = model.attacker(data)
    node_adv, edge_adv = attacker_output["node_key"], attacker_output["edge_key"]
    h_node_adv = model.graph_backs(x_encode, edge_index, node_adv, edge_adv)
    h_graph_adv = model.pool(h_node_adv, batch)

    return node_adv, edge_adv, h_graph_adv

def get_h_DA(model, x_encode, data, node_adv, edge_adv, node_cau, edge_cau):
    x, edge_index, batch = data.x, data.edge_index, data.batch
    node_com = (1 - node_cau) * node_adv + node_cau
    edge_com = (1 - edge_cau) * edge_adv + edge_cau
    h_node_com = model.graph_backs(x_encode, edge_index, node_com, edge_com)
    h_graph_com = model.pool(h_node_com, batch)

    return h_graph_com

def     eval(model, loader, device):
    model.eval()
    correct_cau = 0
    correct_com = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred_cau = model.forward_causal(data).max(1)[1]
            pred_com = model.forward_combined_inference(data).max(1)[1]
        correct_cau += pred_cau.eq(data.y.view(-1)).sum().item()
        correct_com += pred_com.eq(data.y.view(-1)).sum().item()
    correct_cau = correct_cau / len(loader.dataset)
    correct_com = correct_com / len(loader.dataset)
    return correct_cau * 100, correct_com * 100

def get_dataloader(dataset, args):
    train_loader = DataLoader(dataset["train"], batch_size=args.batch_size, shuffle=True)
    valid_loader_ood = DataLoader(dataset["val"], batch_size=args.batch_size, shuffle=False)
    test_loader_ood = DataLoader(dataset["test"], batch_size=args.batch_size, shuffle=False)

    compute_avg_node_edge_count(dataset["train"])
    compute_avg_node_edge_count(dataset["val"])
    compute_avg_node_edge_count(dataset["test"])

    return train_loader, valid_loader_ood, test_loader_ood

def compute_avg_node_edge_count(subset):
    total_nodes = 0
    total_edges = 0

    for data in subset:
        total_nodes += data.num_nodes
        total_edges += data.num_edges

    avg_nodes = total_nodes / len(subset)
    avg_edges = total_edges / len(subset)

    return avg_nodes, avg_edges

