import time

import torch
import tqdm
from torch.utils.tensorboard import SummaryWriter

from GOOD.data.good_datasets.good_motif import GOODMotif
from args import parse_args
from nets.Motif.models import CausalAdvGNNSyn
from nets.Motif.utils import get_h_causaler, get_h_attack, get_h_DA, eval, get_dataloader
from common_utils import set_random_seed, info_nce, info_nce_DA, \
    print_log, get_OPT, get_schedule, set_network_train_eval


def train():
    model.train()
    total_loss_causaler, total_loss_attacker = 0, 0
    causaler_branch = [model.graph_front, model.graph_backs, model.causaler, model.predictor]
    attacker_branch = [model.attacker]
    for step, data in enumerate(train_loader):
        data = data.to(args.device)

        set_network_train_eval(causaler_branch, attacker_branch, train="causaler")

        x_encode = model.graph_front(data.x, data.edge_index)
        node_cau, edge_cau, h_graph_cau, loss_reg = get_h_causaler(model, x_encode, data)
        node_adv, edge_adv, h_graph_adv = get_h_attack(model, x_encode, data)
        h_graph_com = get_h_DA(model, x_encode, data, node_adv, edge_adv, node_cau, edge_cau)

        loss_cl_com = cl_criterion_DA(h_graph_cau, h_graph_com)

        pred_cau = model.predictor(h_graph_cau)
        pred_com = model.predictor(h_graph_com)

        loss_cau_truth = criterion(pred_cau, data.y.long())
        loss_com_truth = criterion(pred_com, data.y.long())


        loss_causaler = loss_cl_com + (loss_cau_truth + loss_com_truth) * args.label_reg


        total_loss_causaler += loss_causaler.item()

        loss_causaler.backward()
        opt_causaler.step()

        set_network_train_eval(causaler_branch, attacker_branch, train="attacker")
        pred_attack = model.forward_attack(data, model)
        loss_adv = criterion(pred_attack["pred_adv"], data.y.long())
        loss_dis = pred_attack["loss_dis"]
        loss_reg = pred_attack["adv_loss_reg"]

        loss_attacker = loss_adv - loss_dis * args.adv_dis - loss_reg * args.adv_reg
        (-loss_attacker).backward()

        opt_attacker.step()
        total_loss_attacker += loss_attacker.item()

    epoch_loss_causaler = total_loss_causaler / len(train_loader)
    # writer.add_scalar("epoch_loss_causaler", epoch_loss_causaler, epoch)




def evaluate():
    valid_cau, valid_com = eval(model, valid_loader_ood, args.device)
    test_cau, test_com = eval(model, test_loader_ood, args.device)

    if valid_cau > results['highest_valid'] and epoch > args.test_epoch:
        results['highest_valid'] = valid_cau
        results['update_test'] = test_cau
        results['update_epoch'] = epoch



    if args.lr_scheduler in ['step', 'muti', 'cos']:
        scheduler_causaler.step()
        scheduler_attacker.step()


final_test_acc = []
args = parse_args()
start_time = time.time()
# writer = None


assert args.dataset == 'GOOD-Motif'

for repetition in range(args.repetition):
    args.seed += 10
    set_random_seed(args.seed)
    # writer = SummaryWriter('logs')
    results = {'highest_valid': 0, 'update_test': 0,  'update_epoch': 0}


    dataset, _ = GOODMotif.load(args.path, domain=args.domain, shift='concept', generate=False)
    train_loader, valid_loader_ood, test_loader_ood = get_dataloader(dataset, args)
    num_class, in_dim = 3, 1

    model = CausalAdvGNNSyn(num_class=num_class,
                            in_dim=in_dim,
                            emb_dim=300,
                            cau_gamma=args.cau_gamma,
                            adv_gamma=args.adv_gamma_node,
                            adv_gamma_edge=args.adv_gamma_edge).to(args.device)
    fine_model = './pretrain_model/concept/AIA_params_Motif_' + args.domain + '.pth'
    model.load_state_dict(torch.load(fine_model))

    opt_attacker, opt_causaler = get_OPT(model, args.lr, args.l2reg)
    scheduler_attacker, scheduler_causaler = get_schedule(opt_attacker, opt_causaler, args)

    cl_criterion_DA = info_nce_DA
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in tqdm.tqdm(range(1, args.epochs + 1)):
        train()
        evaluate()

    final_test_acc.append(results['update_test'])
    # writer.close()

total_time = time.time() - start_time
print_log(args, final_test_acc, total_time, fine_model)








