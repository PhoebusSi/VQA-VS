import torch
import torch.nn as nn
from torch.autograd import Variable
from transformers import LxmertTokenizer, LxmertModel
import numpy as np
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier, PaperClassifier
#from vqa_debias_loss_functions import *

from torch.nn import functional as F
from fc import FCNet, GTH
from attention import Att_0, Att_1, Att_2, Att_3, Att_P, Att_PD, Att_3S
import torch
import random
class Model(nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        self.model = LxmertModel.from_pretrained('unc-nlp/lxmert-base-uncased', return_dict=True)#.cuda()
        self.model = nn.DataParallel(self.model)#.cuda()
        self.batchsize = opt.batch_size
        self.Linear_layer = nn.Linear(768, 1)#.cuda()
        norm = opt.norm#"weight"                                                      
        activation = opt.activation#Relu                               
        dropC = opt.dropC#0.5                                                         
        self.classifier = SimpleClassifier(in_dim=768, hid_dim=2 * 768, out_dim=opt.ans_dim,    
                                           dropout=dropC, norm=norm, act=activation)  

    def forward(self,  q, gv_pos, b):
        """
        qa_text (btachsize, condi_ans_num, max_length)
        v (batchsize, obj_num, v_dim)
        b (batchsize, obj_num, b_dim)

        return: logits
        """
        q = q.cuda()
        batch_size = q.size(0)
        gv_pos= gv_pos.cuda()
        b= b.cuda()

        outputs = self.model(q, gv_pos, b)
        pool_out = outputs.pooled_output
        
        logits = self.classifier(pool_out)


 
        return logits




