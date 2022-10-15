"""
This code is modified from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa
"""
from __future__ import print_function
from collections  import Counter
import os
import math
import json
from transformers import LxmertTokenizer, LxmertModel
import _pickle as cPickle
import numpy as np
import pickle
import utils
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import h5py
from xml.etree.ElementTree import parse
import torch
from torch.utils.data import Dataset
import zarr
import random
COUNTING_ONLY = False





def _create_entry(img, question, answer):
    #print(question.keys())
    #print(answer.keys())

    if None != answer:
        answer.pop('image_id')
        answer.pop('question_id')
        entry = {
            'question_id': question['question_id'],
            'image_id': question['image_id'],
            'image': img,
            'question': question['question'],
            'answer': answer}
    else:
        entry = {
            'question_id': question['question_id'],
            'image_id': question['image_id'],
            'image': img,
            'question': question['question'],
            'answer': None}
    return entry


def get_ques_ans_path(name):
    
    answer_path = os.path.join('./cache/', '%s_target.pkl' % name)
    if name == 'train':
        question_path = './VQA-VS/Training/Training-Ques.json'
    elif name == 'val':
        question_path = './VQA-VS/Val/Val-Ques.json'
    elif name == 'test':
        question_path = './VQA-VS/IID-Test/IID-Test-Ques.json'
    else:
        print('plz set name is one of [train, val, test]!')
        assert 1==2
    if name=='test':
        return question_path
    return question_path, answer_path
 
def _load_dataset(dataroot, name, label2ans,ratio=1.0):
    """Load entries

    img_id2val: dict {img_id -> val} val can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val', 'test'
    """
    if name == "test":
        question_path = get_ques_ans_path(name)
    else:
        question_path, answer_path = get_ques_ans_path(name)
    
    #question_path = os.path.join(dataroot, 'vqacp_v2_%s_questions.json' % (name))
    questions = sorted(json.load(open(question_path)),
                           key=lambda x: x['question_id'])

    # train, val
    #answer_path = os.path.join(dataroot, 'cache', '%s_target.pkl' % name)
    if name != "test":
        answers = cPickle.load(open(answer_path, 'rb'))
        answers = sorted(answers, key=lambda x: x['question_id'])[0:len(questions)]

        utils.assert_eq(len(questions), len(answers))

    if ratio < 1.0:
        # sampling traing instance to construct smaller training set.
        index = random.sample(range(0,len(questions)), int(len(questions)*ratio))
        questions_new = [questions[i] for i in index]
        if name != "test":
            answers_new = [answers[i] for i in index]
    else:
        questions_new = questions
        if name != "test":
            answers_new = answers

    entries = []
    if name != "test":
        for question, answer in zip(questions_new, answers_new):
            utils.assert_eq(question['question_id'], answer['question_id'])
            utils.assert_eq(question['image_id'], answer['image_id'])
            img_id = question['image_id']
            entries.append(_create_entry(img_id, question, answer))
    else:
        for question in questions_new:
            img_id = question['image_id']
            entries.append(_create_entry(img_id, question, None))
    return entries


class VQAFeatureDataset(Dataset):
    def __init__(self, name, dataroot, image_dataroot, ratio, adaptive=False):
        super(VQAFeatureDataset, self).__init__()
        assert name in ['train', 'test', 'val']

        ans2label_path = os.path.join( './cache/', 'train_val_ans2label.pkl')
        label2ans_path = os.path.join( './cache/', 'train_val_label2ans.pkl')
        self.ans2label = cPickle.load(open(ans2label_path, 'rb'))
        self.label2ans = cPickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)

        self.adaptive = adaptive

        print('loading image features and bounding boxes')
        # Load image features and bounding boxes
        print(image_dataroot)
        with open("/root/VQA/data/coco/object_features/vqa_img_feature_trainval.pickle", "rb") as f_f:
            f_f_data = pickle.load(f_f)
            print(len(list(f_f_data.keys())))
            # print("f_f_data", f_f_data)
        self.features =  f_f_data

        print('loading image features and bounding boxes done!')
        self.entries = _load_dataset(dataroot, name, self.label2ans, ratio)
        #这里的entries是一个列表，列表的每一个元素还是一个字典{question_id,image_id,image(等同image_id),question文本，anwer字典{labels,scores}}
        self.tokenize()
        self.tensorize(name)
            
    def tokenize(self, max_length=14):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        tokenizer = LxmertTokenizer.from_pretrained('unc-nlp/lxmert-base-uncased')
        for entry in self.entries:
            question_text = entry['question'] 
            lower_question_text = question_text.lower()
            ques_token_dict = tokenizer(lower_question_text)
            q_tokens = ques_token_dict['input_ids']
            length = len(q_tokens)
 



            if len(q_tokens) > max_length :
                q_tokens = q_tokens[:max_length]
                length = max_length
            else:
                padding = [tokenizer('[PAD]')['input_ids'][1:-1][0]]*(max_length - len(q_tokens))
                q_tokens = q_tokens + padding

            utils.assert_eq(len(q_tokens), max_length)
            entry['q_token'] = q_tokens
            entry['length'] = length
    def tensorize(self, name):
        for entry in self.entries:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question
            length = torch.from_numpy(np.array(entry['length']))
            entry['length'] = length

              
            answer = entry['answer']
            if None != answer:
                labels = np.array(answer['labels'])
                scores = np.array(answer['scores'], dtype=np.float32)
                if len(labels):
                    labels = torch.from_numpy(labels)
                    scores = torch.from_numpy(scores)
                    entry['answer']['labels'] = labels
                    entry['answer']['scores'] = scores
                else:
                    entry['answer']['labels'] = None
                    entry['answer']['scores'] = None



    def __getitem__(self, index):
        entry = self.entries[index]
        if not self.adaptive:
            features = torch.from_numpy(np.array(self.features[str(entry['image'])]['feats']))
            spatials = torch.from_numpy(np.array(self.features[str(entry['image'])]['sp_feats']))
            features = features.to(torch.float32)
            spatials = spatials.to(torch.float32)
        max_length = 14

        question = entry['q_token']
        

        question_id = entry['question_id']
        answer = entry['answer']
        if None != answer:
            labels = answer['labels']
            scores = answer['scores']
            target = torch.zeros(self.num_ans_candidates)
            if labels is not None:
                target.scatter_(0, labels, scores)
            return features, spatials, question, target, question_id
        else:
            return features, spatials, question, question_id

    def __len__(self):
        return len(self.entries)


if __name__ == '__main__':

    from torch.utils.data import DataLoader

    dataroot = '../data/annotations/'
    img_root = '../data/coco/'
    train_dset = VQAFeatureDataset('train', dataroot, img_root, ratio=1.0, adaptive=False)

    loader = DataLoader(train_dset, 256, shuffle=True, num_workers=1, collate_fn=utils.trim_collate)

    for v, b, q, a, qid in loader:

        print(a.shape)
