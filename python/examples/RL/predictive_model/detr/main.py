# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
from pathlib import Path

import numpy as np
import torch
from .models import build_ACT_model, build_CNNMLP_model

import IPython
e = IPython.embed

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float) # will be overridden
    parser.add_argument('--lr_backbone', default=1e-5, type=float) # will be overridden
    parser.add_argument('--batch_size', default=2, type=int) # not used
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int) # not used
    parser.add_argument('--lr_drop', default=200, type=int) # not used
    parser.add_argument('--clip_max_norm', default=0.1, type=float, # not used
                        help='gradient clipping max norm')

    # Model parameters
    # * Backbone
    parser.add_argument('--backbone', default='resnet18', type=str, # will be overridden
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--camera_names', default=[], type=list, # will be overridden
                        help="A list of camera names")

    # * Transformer
    parser.add_argument('--enc_layers', default=4, type=int, # will be overridden
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int, # will be overridden
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int, # will be overridden
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int, # will be overridden
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int, # will be overridden
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', action='store', default=400, type=int, # will be overridden
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # repeat args in imitate_episodes just to avoid error. Will not be used

    parser.add_argument('--eval',               action='store_true')
    parser.add_argument('--onscreen_render',    action='store_true')
    parser.add_argument('--ckpt_dir',           action='store', type=str,   required=False,                      help='ckpt_dir')
    parser.add_argument('--policy_class',       action='store', type=str,   required=False,      default="ACT",  help='policy_class, capitalize')
    parser.add_argument('--task_name',          action='store', type=str,   required=False,                      help='task_name')
    # parser.add_argument('--batch_size',         action='store', type=int,   required=True,                      help='batch_size')
    parser.add_argument('--seed',               action='store', type=int,   required=False,                      help='seed')
    parser.add_argument('--num_steps',          action='store', type=int,   required=False,                      help='num_steps')
    # parser.add_argument('--lr',                 action='store', type=float, required=True,                      help='lr')
    parser.add_argument('--load_pretrain',      action='store_true',                            default=False)
    parser.add_argument('--eval_every',         action='store', type=int,   required=False,     default=100000, help='eval_every', )
    parser.add_argument('--validate_every',     action='store', type=int,   required=False,     default=2500,   help='validate_every', )
    parser.add_argument('--save_every',         action='store', type=int,   required=False,     default=2500,   help='save_every', )
    parser.add_argument('--resume_ckpt_path',   action='store', type=str,   required=False,                     help='resume_ckpt_path', )
    parser.add_argument('--skip_mirrored_data', action='store_true')                      ,     
    parser.add_argument('--actuator_network_dir', action='store', type=str, required=False,                     help='actuator_network_dir', )
    parser.add_argument('--history_len',        action='store', type=int)
    parser.add_argument('--future_len',         action='store', type=int)
    parser.add_argument('--prediction_len',     action='store', type=int)

    # for ACT
    parser.add_argument('--kl_weight',          action='store', type=int,   required=False,     default=10,     help='KL Weight',       )
    parser.add_argument('--chunk_size',         action='store', type=int,   required=False,     default= 25,     help='chunk_size',      )
    # parser.add_argument('--hidden_dim',         action='store', type=int,   required=False,     default=512,    help='hidden_dim',      )
    # parser.add_argument('--dim_feedforward',    action='store', type=int,   required=False,     default=2048,   help='dim_feedforward', )
    parser.add_argument('--temporal_agg',       action='store_true')
    parser.add_argument('--use_vq',             action='store_true')
    parser.add_argument('--vq_class',           action='store', type=int,   help='vq_class')
    parser.add_argument('--vq_dim',             action='store', type=int,   help='vq_dim')
    parser.add_argument('--no_encoder',         action='store_true')
    
    # for Isaacgym
    parser.add_argument('--num_envs',          action='store', type=int,   required=False,     default=1,     help='KL Weight',       )
    parser.add_argument('--num_box',          action='store', type=int,   required=False,     default=1,     help='KL Weight',       )
    parser.add_argument('--sim_device',          action='store', type=str,   required=False,     default="cpu",           )
    parser.add_argument('--headless',          action='store', type=bool,   required=False,     default=False,           )
    parser.add_argument('--record',          action='store',  type=bool,    default= False,     help= "Set random seed for the simulation" )
    parser.add_argument('--alpha',          action='store',  type=float,    default= 0.0,     help= "Set random seed for the simulation" )
    parser.add_argument('--topic',          action='store', type=str,   required=False,     )
    parser.add_argument('--temp',          action='store',  type=bool,    default= False,     help= "Set random seed for the simulation" )
    
    return parser


def build_ACT_model_and_optimizer(args_override):
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    for k, v in args_override.items():
        setattr(args, k, v)

    model = build_ACT_model(args)
    model.cuda()

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)

    return model, optimizer


def build_CNNMLP_model_and_optimizer(args_override):
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    for k, v in args_override.items():
        setattr(args, k, v)

    model = build_CNNMLP_model(args)
    model.cuda()

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)

    return model, optimizer

