import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    parser.add_argument('--seed', type=int, default=156, help='seed')
    parser.add_argument('--dataset', type=str, default='GOOD-CMNIST',
    choices=['ogbg-molbbbp, ogbg-molbace, GOOD-HIV, GOOD-CMNIST, GOOD-Motif'])
    parser.add_argument('--path', type=str, default='./datasets/', help='path')
    parser.add_argument('--epochs', type=int, default=100, help='epoch')
    parser.add_argument('--test_epoch', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--l2reg', type=float, default=5e-6, help='L2 norm')
    parser.add_argument('--repetition', type=int, default=1, help='repetition time of algorithm')
    parser.add_argument('--domain', type=str, default='color', choices=['size', 'scaffold', 'basis'],
                        help='Molbbbp: [size, scaffold],'
                             'Motif: [size, basis],'
                             'MolHiv: [scaffold, size],'
                             'CMNIST: [color],'
                             'Molbace: [size, scaffold]',)
    parser.add_argument('--size', type=str, default='ls',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')

    # schedule setting
    parser.add_argument('--lr_scheduler', type=str, default='cos', help='')
    parser.add_argument('--lr_gamma', type=float, default=0.1)
    parser.add_argument('--lr_decay', type=int, default=150)
    parser.add_argument('--lr_min', type=float, default=1e-7)
    parser.add_argument('--milestones', nargs='+', type=int, default=[3703, 16, 6])

    #model setting
    parser.add_argument('--emb_dim', type=int, default=300)
    parser.add_argument('--cau_gamma', type=float, default=0.4)
    parser.add_argument('--cau_reg', type=float, default=0.05)
    parser.add_argument('--adv_gamma_node', type=float, default=1.0)
    parser.add_argument('--adv_gamma_edge', type=float, default=1.0)
    parser.add_argument('--cl_rate', type=float, default=0.1)
    parser.add_argument('--adv_dis', type=float, default=0.5)
    parser.add_argument('--adv_reg', type=float, default=0.05)
    parser.add_argument('--fro_layer', type=int, default=2)

    parser.add_argument('--label_reg', type=int, default=1)



    args = parser.parse_args()

    return args
