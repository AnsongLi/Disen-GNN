import argparse
import pickle
import time
from utils import build_graph, Data, split_validation
from model import *
import random
import os
import time
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='diginetica', help='dataset name: diginetica/yoochoose1_4/yoochoose1_64/sample/Tmall/Nowplaying/RetailRocket')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=80, help='hidden state size')
parser.add_argument('--epoch', type=int, default=7, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.005, help='learning rate')  # [0.005, 0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
parser.add_argument('--patience', type=int, default=10, help='the number of epoch to wait before early stop ')
parser.add_argument('--nonhybrid', action='store_true', help='only use the global preference to predict')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')
parser.add_argument('--beta', type=float, default=0.1, help='beta in DisConv')
parser.add_argument('--iterations', type=int, default=2, help='iterations of each DisConv')
parser.add_argument('--score_channel', type=int, default=2, help='should < channel')
parser.add_argument('--dropout', action='store', default=0.1, type=float, help='dropout')
parser.add_argument('--corDecay', type=int, default=5,help='Distance Correlation Weight')
parser.add_argument('--weightshare', type=bool, default=False,help='wether layer share weight')#表示不同的解纠缠层嵌入空间是否一致；当为True时，通道数和维度数都必须要相同；当为False时，保持总维度相同.
# 本文所有实验在两层的解纠缠层中嵌入空间不同。
parser.add_argument('--seed', type=int, default=66,help='random seed')
parser.add_argument('--numcuda', type=int, default=0,help='which GPU train')
opt = parser.parse_args()
opt.layer_num = 2
opt.channels = [5 ,5]#每一层的通道数
opt.c_dims = [16,16]#每一层通道的size
opt.in_dims = [80,80]#，每一层输入的hiddensize 无论如何都要保证每一层的hiddensize相同
print(opt)

def seed_torch(seed):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

def main():
    if torch.cuda.is_available():
        seed_torch(opt.seed)
        torch.cuda.set_device(opt.numcuda)
    train_data = pickle.load(open('../datasets/' + opt.dataset + '/train.txt', 'rb'))
    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open('../datasets/' + opt.dataset + '/test.txt', 'rb'))
    if opt.dataset == 'diginetica':
        n_node = 43098
    elif opt.dataset == 'yoochoose1_64' or opt.dataset == 'yoochoose1_4':
        n_node = 37484
    elif opt.dataset == 'RetailRocket':
        n_node = 36969
    elif opt.dataset == 'Tmall' :
        n_node = 40728
    elif opt.dataset == 'Nowplaying':
        n_node = 60417
    else:
        n_node = 310


    train_data = Data(train_data, shuffle=True)
    test_data = Data(test_data, shuffle=False)
    model = trans_to_cuda(SessionGraph(opt, n_node))
    Cor_loss_model = trans_to_cuda(Cor_loss(opt.corDecay,opt.channels[-1],opt.in_dims[-1],opt.c_dims[-1]))



    start = time.time()
    best_result = [0, 0]
    best_epoch = [0, 0]
    bad_counter = 0
    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        hit, mrr = train_test_cor(model, train_data, test_data,Cor_loss_model)
        flag = 0
        if hit >= best_result[0]:
            best_result[0] = hit
            best_epoch[0] = epoch
            flag = 1
        if mrr >= best_result[1]:
            best_result[1] = mrr
            best_epoch[1] = epoch
            flag = 1
        print('Best Result:')
        print('\tRecall@20:\t%.4f\tMMR@20:\t%.4f\tEpoch:\t%d,\t%d'% (best_result[0], best_result[1], best_epoch[0], best_epoch[1]))
        bad_counter += 1 - flag
        if bad_counter >= opt.patience:
            break
    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))








if __name__ == '__main__':
    main()
