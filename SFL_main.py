import timm
import os
from pandas import DataFrame
import matplotlib
import torch
import numpy as np
from torch import nn
import random
from torch.utils.data import Subset, DataLoader
from dataset import Cifar10Set,create_non_iid_data_distribution,Cifar100Set
matplotlib.use('Agg')
from torch.optim import SGD
import copy
from SLmodels.FedAvg import FedAvg
from args import args_parser
from SLmodels.sflmodels import ViT_server_side,ViT_client_side
def prRed(skk): print("\033[91m {}\033[00m".format(skk))
def prGreen(skk): print("\033[92m {}\033[00m".format(skk))
args = args_parser()
SEED = args.seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    print(torch.cuda.get_device_name(0))

program = f"SFLV1_{args.model_name}_{args.pretrained}_{args.dataset_name}ds_{args.Rounds}r_{args.local_epoch}ep_{args.lr}lr_{args.batch_size}bs_{args.alpha}a_{args.num_selected_clients}client_{args.bit_width}bit"
print(f"---------{program}----------")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calculate_accuracy(fx, y):
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = 100.00 * correct.float() / preds.shape[0]
    return acc

def train_server(fx_client, y, l_epoch_count, l_epoch, idx, len_batch):
    global net_glob_server, criterion, device, batch_acc_train, batch_loss_train, l_epoch_check, fed_check
    global loss_train_collect, acc_train_collect, count1, acc_avg_all_user_train, loss_avg_all_user_train, idx_collect
    global loss_train_collect_user, acc_train_collect_user

    net_glob_server.train()
    optimizer_server = SGD(net_glob_server.parameters(), lr=args.lr, momentum=0.9)#torch.optim.Adam(net_glob_server.parameters(), lr=args.lr)
    optimizer_server.zero_grad()

    fx_client = fx_client.to(device)
    y = y.to(device)
    fx_server = net_glob_server(fx_client.to(torch.float))

    loss = criterion(fx_server, y)
    acc = calculate_accuracy(fx_server, y)

    loss.backward()
    dfx_client = fx_client.grad.clone().detach()
    optimizer_server.step()

    batch_loss_train.append(loss.item())
    batch_acc_train.append(acc.item())

    count1 += 1
    if count1 == len_batch:
        acc_avg_train = sum(batch_acc_train) / len(batch_acc_train)  # it has accuracy for one batch
        loss_avg_train = sum(batch_loss_train) / len(batch_loss_train)

        batch_acc_train = []
        batch_loss_train = []
        count1 = 0

        prRed('Client{} Train => Local Epoch: {} \tAcc: {:.3f} \tLoss: {:.4f}'.format(idx, l_epoch_count, acc_avg_train,
                                                                                      loss_avg_train))

        if l_epoch_count == l_epoch - 1:

            l_epoch_check = True
            acc_avg_train_all = acc_avg_train
            loss_avg_train_all = loss_avg_train
            loss_train_collect_user.append(loss_avg_train_all)
            acc_train_collect_user.append(acc_avg_train_all)
            if idx not in idx_collect:
                idx_collect.append(idx)

        global selcted_idx_users
        if len(idx_collect) == len(selcted_idx_users):
            fed_check = True
            idx_collect = []
            acc_avg_all_user_train = sum(acc_train_collect_user) / len(acc_train_collect_user)
            loss_avg_all_user_train = sum(loss_train_collect_user) / len(loss_train_collect_user)
            loss_train_collect.append(loss_avg_all_user_train)
            acc_train_collect.append(acc_avg_all_user_train)
            acc_train_collect_user = []
            loss_train_collect_user = []

    return dfx_client

def evaluate_server(fx_client, y, idx, len_batch, ell):
    global net_glob_server, criterion, batch_acc_test, batch_loss_test
    global loss_test_collect, acc_test_collect, count2, acc_avg_train_all, loss_avg_train_all, l_epoch_check, fed_check
    global loss_test_collect_user, acc_test_collect_user, acc_avg_all_user_train, loss_avg_all_user_train

    net_glob_server.eval()

    with torch.no_grad():
        fx_client = fx_client.to(device)
        y = y.to(device)
        fx_server = net_glob_server(fx_client)
        loss = criterion(fx_server, y)
        acc = calculate_accuracy(fx_server, y)
        batch_loss_test.append(loss.item())
        batch_acc_test.append(acc.item())

        count2 += 1
        if count2 == len_batch:
            acc_avg_test = sum(batch_acc_test) / len(batch_acc_test)
            loss_avg_test = sum(batch_loss_test) / len(batch_loss_test)
            batch_acc_test = []
            batch_loss_test = []
            count2 = 0
            prGreen('Client{} Test =>                   \tAcc: {:.3f} \tLoss: {:.4f}'.format(idx, acc_avg_test,
                                                                                             loss_avg_test))
            if l_epoch_check:
                l_epoch_check = False
                acc_avg_test_all = acc_avg_test
                loss_avg_test_all = loss_avg_test
                loss_test_collect_user.append(loss_avg_test_all)
                acc_test_collect_user.append(acc_avg_test_all)

            if fed_check:
                fed_check = False
                acc_avg_all_user = sum(acc_test_collect_user) / len(acc_test_collect_user)
                loss_avg_all_user = sum(loss_test_collect_user) / len(loss_test_collect_user)

                loss_test_collect.append(loss_avg_all_user)
                acc_test_collect.append(acc_avg_all_user)
                acc_test_collect_user = []
                loss_test_collect_user = []

                print("====================== SERVER V1==========================")
                print(' Train: Round {:3d}, Avg Accuracy {:.3f} | Avg Loss {:.3f}'.format(ell, acc_avg_all_user_train,
                                                                                          loss_avg_all_user_train))
                print(' Test: Round {:3d}, Avg Accuracy {:.3f} | Avg Loss {:.3f}'.format(ell, acc_avg_all_user,
                                                                                     loss_avg_all_user))
                print("==========================================================")
    return

class Client(object):
    def __init__(self, idx, lr, device, user_trainset=None, testing_set=None, local_ep=1):
        self.idx = idx
        self.device = device
        self.lr = lr
        self.local_ep = local_ep
        self.selected = False
        self.ldr_train = DataLoader(user_trainset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers) #DataLoader(user_trainset, batch_size=args.batch_size, shuffle=True)
        self.ldr_test = DataLoader(testing_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    def train(self, net):
        net.train()
        optimizer_client = SGD(net.parameters(), lr=self.lr, momentum=0.9)# torch.optim.Adam(net.parameters(), lr=self.lr)

        for iter in range(self.local_ep):
            len_batch = len(self.ldr_train)
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer_client.zero_grad()
                fx = net(images)
                # quantized_fx = quantization(fx, bit_width=args.bit_width)
                # client_fx = quantized_fx.clone().detach().requires_grad_(True)
                client_fx = fx.clone().detach().requires_grad_(True)
                dfx = train_server(client_fx, labels, iter, self.local_ep, self.idx, len_batch)
                fx.backward(dfx)
                optimizer_client.step()
        return net.state_dict()

    def evaluate(self, net, ell):
        net.eval()
        with torch.no_grad():
            len_batch = len(self.ldr_test)
            for batch_idx, (images, labels) in enumerate(self.ldr_test):
                images, labels = images.to(self.device), labels.to(self.device)
                fx = net(images)
                evaluate_server(fx, labels, self.idx, len_batch, ell)
        return

if __name__ == '__main__':
    if args.dataset_name == 'cifar10':
        num_classes = 10
        training_set, testing_set = Cifar10Set(input_size= args.input_size, batch_size = args.batch_size, base_dir= args.base_dir, num_workers=args.num_workers)
        num_channels = 3
    elif args.dataset_name == 'cifar100':
        num_classes = 100
        training_set, testing_set = Cifar100Set(input_size= args.input_size, batch_size = args.batch_size, base_dir= args.base_dir, num_workers=args.num_workers)
        num_channels = 3
    user_data_indices = create_non_iid_data_distribution(args.alpha, args.num_clients, args.dataset_name, training_set)
    vit = timm.create_model(
        model_name = args.model_name,
        pretrained = args.pretrained,
        num_classes = num_classes,
        in_chans=3
    )

    net_glob_client = ViT_client_side(embedding_layer=vit.patch_embed).to(device)
    net_glob_client.to(device)

    net_glob_server = ViT_server_side(vit=vit,num_classes=num_classes).to(device)
    net_glob_server.to(device)

    ini_client_model_dict = copy.deepcopy(net_glob_client.state_dict())
    ini_server_model_dict = copy.deepcopy(net_glob_server.state_dict())

    loss_train_collect = []
    acc_train_collect = []
    loss_test_collect = []
    acc_test_collect = []
    batch_acc_train = []
    batch_loss_train = []
    batch_acc_test = []
    batch_loss_test = []

    criterion = nn.CrossEntropyLoss()
    count1 = 0
    count2 = 0

    acc_avg_all_user_train = 0
    loss_avg_all_user_train = 0
    loss_train_collect_user = []
    acc_train_collect_user = []
    loss_test_collect_user = []
    acc_test_collect_user = []

    idx_collect = []
    l_epoch_check = False
    fed_check = False

    model = vit.to(device)
    client_weights = []
    server_weights = []

    for iter in range(args.Rounds):
        selcted_idx_users = np.random.choice(range(args.num_clients), args.num_selected_clients, replace=False)
        whole_model_list = []
        if len(selcted_idx_users) == 0:
            continue
        client_weights = []
        server_weights = []
        datasize_lists = []
        for idx in selcted_idx_users:
            user_trainset = Subset(training_set, user_data_indices[idx])
            datasize_lists.append(len(user_data_indices[idx]))
            net_glob_client.load_state_dict(copy.deepcopy(ini_client_model_dict))
            net_glob_server.load_state_dict(copy.deepcopy(ini_server_model_dict))
            local = Client(idx, args.lr, device, user_trainset, testing_set, local_ep=args.local_epoch)
            w_client = local.train(net=copy.deepcopy(net_glob_client).to(device))
            local.evaluate(net=copy.deepcopy(net_glob_client).to(device), ell=iter)
            client_weights.append(w_client)
            server_weights.append(net_glob_server.state_dict())

        ini_client_model_dict = FedAvg(client_weights,datasize_lists)
        ini_server_model_dict = FedAvg(server_weights,datasize_lists)
        print(f"Allocated memory: {torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB")
        torch.cuda.empty_cache()

        # ===================================================================================
        print("Training and Evaluation completed!")
        # ===============================================================================
        round_process = [i for i in range(1, len(acc_train_collect) + 1)]

        df = DataFrame({
            'round': round_process,
            'acc_train': acc_train_collect,
            'acc_test': acc_test_collect,
            "loss_train": loss_train_collect,
            "loss_test": loss_test_collect
        })

        print(df)
        file_name = program + ".csv"
        output_dir = "./result/"
        os.makedirs(output_dir, exist_ok=True)  # 确保输出目录存在
        df.to_csv(os.path.join(output_dir, file_name), index=False)


