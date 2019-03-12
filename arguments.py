#arguments.py
#list arguments being passed to parser
# lr : 0.001
# batch_size : 128
# momentum : 0.9
# weight_decay : 0.0001
# ip_dim : 2
# hl_dim : 10
# op_dim : 2

import argparse
from config import *
parser = argparse.ArgumentParser(description='PyTorch Boilerplate code')
parser.add_argument('--lr', '--learning-rate', default=config['lr'], type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=config['momentum'], type=float, 
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=config['weight_decay'], type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--bs', '--batch_size', default=config['batch_size'], type=int,
                    help='Batch Size')
parser.add_argument('--ip_dim', default=config['ip_dim'], type=int,
                    help='ip_dim')
parser.add_argument('--hl_dim', default=config['hl_dim'], type=int,
                    help='hl_dim')
parser.add_argument('--op_dim', default=config['op_dim'], type=int,
                    help='op_dim')                
parser.add_argument('--device', default=config['device'], type=str,
                    help='op_dim')                             
args = parser.parse_args()
args = vars(args)