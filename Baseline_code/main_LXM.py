import sys
import random
import torch
from collections import defaultdict, Counter
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.init as init
import numpy as np

from dataset_LXM_vqacp import VQAFeatureDataset


from lxmert_model_3000 import Model
import utils
import opts_LXM as opts
from train_LXM import train


def weights_init_kn(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data, a=0.01)


if __name__ == '__main__':
    opt = opts.parse_opt()
    seed = 0
    if opt.seed == 0:
        seed = random.randint(1, 10000)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(opt.seed)
    else:
        seed = opt.seed
        random.seed(seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = True



    model = Model(opt)
    model = model.cuda()

    train_dset = VQAFeatureDataset('train', opt.dataroot, opt.img_root, ratio=opt.ratio, adaptive=False)  # load labeld data

    
    val_dset = VQAFeatureDataset('val', opt.dataroot, opt.img_root,ratio=1.0, adaptive=False)


    train_loader = DataLoader(train_dset, opt.batch_size, shuffle=True, num_workers=1, collate_fn=utils.trim_collate)
    opt.use_all = 1
    

    val_loader = DataLoader(val_dset, opt.batch_size, shuffle=False, num_workers=1, collate_fn=utils.trim_collate)

    
    train(model, train_loader, val_loader, opt)








