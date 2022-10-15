import os
import random
import time
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import json
from torch.optim.lr_scheduler import MultiStepLR, ConstantLR


# standard cross-entropy loss
def instance_bce(logits, labels):
    assert logits.dim() == 2
    cross_entropy_loss = nn.CrossEntropyLoss()

    prediction_ans_k, top_ans_ind = torch.topk(F.softmax(labels, dim=-1), k=1, dim=-1, sorted=False)
    ce_loss = cross_entropy_loss(logits, top_ans_ind.squeeze(-1))

    return ce_loss

# multi-label soft loss
def instance_bce_with_logits(logits, labels, reduction='mean'):
    assert logits.dim() == 2
    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction=reduction)
    if reduction == 'mean':
        loss *= labels.size(1)
    return loss

def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data  # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores



def multi_set_evaluation(model, set_list):
    final_score = []
    final_bound = []
    for eval_loader in set_list:
        eval_score, bound = evaluate(model, eval_loader)
        final_score.append(eval_score)
        final_bound.append(bound)
        #final_entropy.append(entropy)
    return final_score, final_bound


def  train(model, train_loader, val_loader, opt):
    
    utils.create_dir(opt.output)
    optim = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, betas=(0.9, 0.999), eps=1e-08,
                             weight_decay=opt.weight_decay)
    logger = utils.Logger(os.path.join(opt.output, 'VQA-VS-LXMERT_log.txt'))

    utils.print_model(model, logger)

    # load snapshot
    if opt.checkpoint_path is not None:
        print('loading %s' % opt.checkpoint_path)
        model_data = torch.load(opt.checkpoint_path)
        model.load_state_dict(model_data.get('model_state', model_data))
        optim.load_state_dict(model_data.get('optimizer_state', model_data))
        opt.s_epoch = model_data['epoch'] + 1

    for param_group in optim.param_groups:
        param_group['lr'] = opt.learning_rate
    
    scheduler = ConstantLR(optim)

    


    val_best_score = 0
    for epoch in range(opt.s_epoch, opt.num_epochs):
        total_loss = 0
        total_bce_loss = 0
        train_score = 0
        total_norm = 0
        count_norm = 0
        t = time.time()
        N = len(train_loader.dataset)
        scheduler.step()

        for i, (v, b, q, a,_) in enumerate(train_loader):
            #print(q)
            v = v.cuda()
            q = q.cuda()
            b = b.cuda()
            a = a.cuda()
            # for the labeled samples

    
            logits = model(q, v, b, a)
            bce_loss = instance_bce_with_logits(logits, a, reduction='mean')
            loss = bce_loss

            loss.backward()
            total_norm += nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
            count_norm += 1
            optim.step()
            optim.zero_grad()

            score_pos = compute_score_with_logits(logits, a.data).sum()
            train_score += score_pos.item()
            total_loss += loss.item() * v.size(0)
            total_bce_loss += bce_loss.item() * v.size(0)
            

            if i != 0 and i % 100 == 0:
                print(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()) )
                print(
                    'traing: %d/%d, train_loss: %.6f, bce_loss: %.6f, train_acc: %.6f' %
                    (i, len(train_loader), total_loss / (i * v.size(0)),
                     total_bce_loss / (i * v.size(0)),
                     100 * train_score / (i * v.size(0))))

        total_loss /= N
        total_bce_loss /= N
        train_score = 100 * train_score / N
        
        model.train(False)
        val_score, val_bound = evaluate(model, val_loader)
        model.train(True)
        # logger.write('\nlr: %.7f' % optim.param_groups[0]['lr'])
        logger.write('epoch %d, time: %.2f' % (epoch, time.time() - t))
        logger.write(
            '\ttrain_loss: %.2f, norm: %.4f, score: %.2f' % (total_loss, total_norm / count_norm, train_score))
        
        logger.write('\t: val_score:  %.2f' % (100 * val_score))
        logger.write('\t: val_bound:  %.2f' % (100 * val_bound))

        if  val_score > val_best_score:
            model_path = os.path.join(opt.output, 'best_model.pth')
            utils.save_model(model_path, model, epoch, optim)
            val_best_score = val_score


@torch.no_grad()
def evaluate(model, dataloader):
    score = 0
    upper_bound = 0
    num_data = 0
    for i, (v, b, q, a, q_id) in enumerate(dataloader):
        #print(q)
        v = v.cuda()
        b = b.cuda()
        q = q.cuda()
        a = a.cuda()
        q_id = q_id.cuda()
        pred = model(q, v, b, a)
        batch_score = compute_score_with_logits(pred, a.cuda()).sum()
        score += batch_score.item()
        upper_bound += (a.max(1)[0]).sum().item()
        num_data += pred.size(0)


    score = score / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)



    return score, upper_bound



