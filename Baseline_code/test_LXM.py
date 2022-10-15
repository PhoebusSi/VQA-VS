"""
This code is modified from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa
"""
import argparse
import json
import progressbar
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from dataset_LXM_vqacp import VQAFeatureDataset
from lxmert_model_3000 import Model
import utils
import opts_LXM as opts 



def get_question(q, dataloader):
    str = []
    dictionary = dataloader.dataset.dictionary
    for i in range(q.size(0)):
        str.append(dictionary.idx2word[q[i]] if q[i] < len(dictionary.idx2word) else '_')
    return ' '.join(str)


def get_answer(p, dataloader):
    _m, idx = p.max(0)
    return dataloader.dataset.label2ans[idx.item()]


@torch.no_grad()
def get_logits(model, dataloader):
    N = len(dataloader.dataset)
    M = dataloader.dataset.num_ans_candidates
    K = 36
    pred = torch.FloatTensor(N, M).zero_()
    qIds = torch.IntTensor(N).zero_()
    idx = 0
    bar = progressbar.ProgressBar(maxval=N or None).start()
    for v, b, q, i in iter(dataloader):
        bar.update(idx)
        batch_size = v.size(0)
        v = v.cuda()
        b = b.cuda()
        q = q.cuda()
        
        logits = model(q, v, b)
        pred[idx:idx+batch_size,:].copy_(logits.data)
        qIds[idx:idx+batch_size].copy_(i)
        idx += batch_size

    bar.update(idx)
    return pred, qIds


def make_json(logits, qIds, dataloader):
    utils.assert_eq(logits.size(0), len(qIds))
 
    results = []
    for i in range(logits.size(0)):
        result = {}
        result['question_id'] = qIds[i].item()
        result['answer'] = get_answer(logits[i], dataloader)
        results.append(result)
    return results

if __name__ == '__main__':
    opt = opts.parse_opt()

    torch.backends.cudnn.benchmark = True

    test_dset = VQAFeatureDataset('test', opt.dataroot, opt.img_root,ratio=1.0, adaptive=False)
    

    n_device = torch.cuda.device_count()
    batch_size = opt.batch_size * n_device


    model = Model(opt)
    model = model.cuda()

    test_loader = DataLoader(test_dset, opt.batch_size, shuffle=False, num_workers=1, collate_fn=utils.trim_collate)


    def dump_result(opt, model, model_label, eval_loader, list_name):
        n = 0
        if 0 <= opt.s_epoch:
            model_label += '_epoch%d' % opt.s_epoch
        n+=1
        logits, qIds = get_logits(model, eval_loader)
        results = make_json(logits, qIds, eval_loader)
        
        if opt.logits:
            utils.create_dir('logits/'+model_label)
            torch.save(logits, 'logits/'+model_label+'/'+list_name+str(n)+'logits%d.pth' % opt.s_epoch)
        
        utils.create_dir(opt.output)


        with open(opt.output+'/'+list_name+str(n)+'%s.json' \
            % (model_label), 'w') as f:
            json.dump(results, f)

    def process(args, model, checkpoint_path, model_label, test_loader):

        opt.checkpoint_path = checkpoint_path
        print('loading %s' % opt.checkpoint_path)
        model_data = torch.load(opt.checkpoint_path)

        model.load_state_dict(model_data.get('model_state', model_data))
        model = nn.DataParallel(model).cuda()
        opt.s_epoch = model_data['epoch'] + 1

        model.train(False)
        dump_result(opt, model, model_label, test_loader, 'test')

    checkpoint_path = './saved_models/LXMERT/best_model.pth'
    process(opt, model, checkpoint_path, 'LXMERT',  test_loader)
