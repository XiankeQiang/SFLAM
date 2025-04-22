# -*- coding:utf-8 -*-
"""
@Time: 2023/12/4 10:58
@Author: XiankeQiang
@File: args.py
"""
import argparse
import argparse
def args_parser():
    parser = argparse.ArgumentParser(description="Train an object detector")
    parser.add_argument('--dataset_name', type=str,  default='cifar10',choices=['cifar10', 'cifar100'], help='Dataset Name')
    parser.add_argument('--alpha', type=float, default=0.5, help='Dirichlet alpha')
    parser.add_argument('--input_size',  type=int, default= 224, help='Input size --> (input_size, input_size), default : 224')
    parser.add_argument('--num_workers',  type=int, default= 4, help='Number of workers for dataloaders, default : 4')
    parser.add_argument('--num_clients',  type=int, default= 50, help='Number of Clients, default : 50')
    parser.add_argument('--num_selected_clients', type=int, default=10, help='Number of Selected Clients, default : 10')
    parser.add_argument('--model_name', type=str, default= 'vit_base_patch32_224', help='Model name from timm library, default: vit_base_patch32_224')
    parser.add_argument('--pretrained', type=bool, default= True, help='Pretrained weights flag, default: False')
    parser.add_argument('--batch_size',  type=int, default= 128, help='Batch size, default : 128')
    parser.add_argument('--Rounds',  type=int, default= 100, help='Number of Rounds, default : 100')
    parser.add_argument('--lr',  type=float, default= 1e-2, help='Learning rate, default : 1e-2')
    parser.add_argument('--save_every_epochs',  type=int, default= 10, help='Save metrics every this number of epochs, default: 10')
    parser.add_argument('--seed',  type=int, default= 105, help='Seed, default: 105')
    parser.add_argument('--base_dir', type=str, default= "./data", help='')
    parser.add_argument('--root_dir', type=str, default= None, help='')
    parser.add_argument('--csv_file_path', type=str, default=None, help='')
    parser.add_argument('--bit_width', type=int, default=None, help='')
    parser.add_argument("--local_epoch", type=int, default=5, help='Local round before federation, default : 5')
    args = parser.parse_args()
    return args
