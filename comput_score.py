from os import path as osp
import json
import os
import torch
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/root/VQA/baselines/saved_models/LXMERT/test1LXMERT_epoch33.json')
    parser.add_argument('--name', type=str, default='test')
    parser.add_argument('--dataroot', type=str, default='/root/VQA/baselines/cache')

    args = parser.parse_args()
    return args

def get_scores(annotations, predictions):
	score = 0
	count = 0
	other_score = 0
	yes_no_score = 0
	num_score = 0
	yes_count = 0
	other_count = 0
	num_count = 0
	upper_bound = 0
	upper_bound_num = 0
	upper_bound_yes_no = 0
	upper_bound_other = 0

	for pred, anno in zip(predictions, annotations):
		if pred['question_id'] == anno['question_id']:
			G_T= max(anno['answer_count'].values())
			upper_bound += min(1, G_T / 3)
			if pred['answer'] in anno['answers_word']:
				proba = anno['answer_count'][pred['answer']]
				score += min(1, proba / 3)
				count +=1
				if anno['answer_type'] == 'yes/no':
					yes_no_score += min(1, proba / 3)
					upper_bound_yes_no += min(1, G_T / 3)
					yes_count +=1
				if anno['answer_type'] == 'other':
					other_score += min(1, proba / 3)
					upper_bound_other += min(1, G_T / 3)
					other_count +=1
				if anno['answer_type'] == 'number':
					num_score += min(1, proba / 3)
					upper_bound_num += min(1, G_T / 3)
					num_count +=1
			else:
				score += 0
				yes_no_score +=0
				other_score +=0
				num_score +=0
				if anno['answer_type'] == 'yes/no':
					upper_bound_yes_no += min(1, G_T / 3)
					yes_count +=1
				if anno['answer_type'] == 'other':
					upper_bound_other += min(1, G_T / 3)
					other_count +=1
				if anno['answer_type'] == 'number':
					upper_bound_num += min(1, G_T / 3)
					num_count +=1

	
	print('count:', count, ' score:', round(score*100/len(annotations),2))
	print('Yes/No:', round(100*yes_no_score/yes_count,2), 'Num:', round(100*num_score/num_count,2),
		  'other:', round(100*other_score/other_count,2))

	print('count:', len(annotations), ' upper_bound:', round(score*upper_bound/len(annotations)),2)
	print('upper_bound_Yes/No:', round(100*upper_bound_yes_no/yes_count,2), 'upper_bound_Num:',
		  round(100 * upper_bound_num/num_count,2), 'upper_bound_other:', round(100*upper_bound_other/other_count,2))
	
	print("-------------------------")
	return round(score*100/len(annotations),2)

def get_qid():
	QT_file = json.load(open("/root/VQA/baselines/VQA-VS/OOD-Test/QT/OOD-Test-QT-Ques.json", "r"))
	QT_qid = [k['question_id'] for k in QT_file]
	KW_file = json.load(open("/root/VQA/baselines/VQA-VS/OOD-Test/KW/OOD-Test-KW-Ques.json", "r"))
	KW_qid = [k['question_id'] for k in KW_file]
	KWP_file = json.load(open("/root/VQA/baselines/VQA-VS/OOD-Test/KWP/OOD-Test-KWP-Ques.json", "r"))
	KWP_qid = [k['question_id'] for k in KWP_file]
	QTKW_file = json.load(open("/root/VQA/baselines/VQA-VS/OOD-Test/QT+KW/OOD-Test-QT+KW-Ques.json", "r"))
	QTKW_qid = [k['question_id'] for k in QTKW_file]
	KO_file = json.load(open("/root/VQA/baselines/VQA-VS/OOD-Test/KO/OOD-Test-KO-Ques.json", "r"))
	KO_qid = [k['question_id'] for k in KO_file]
	KOP_file = json.load(open("/root/VQA/baselines/VQA-VS/OOD-Test/KOP/OOD-Test-KOP-Ques.json", "r"))
	KOP_qid = [k['question_id'] for k in KOP_file]
	QTKO_file = json.load(open("/root/VQA/baselines/VQA-VS/OOD-Test/QT+KO/OOD-Test-QT+KO-Ques.json", "r"))
	QTKO_qid = [k['question_id'] for k in QTKO_file]
	KWKO_file = json.load(open("/root/VQA/baselines/VQA-VS/OOD-Test/KW+KO/OOD-Test-KW+KO-Ques.json", "r"))
	KWKO_qid = [k['question_id'] for k in KWKO_file]
	QTKWKO_file = json.load(open("/root/VQA/baselines/VQA-VS/OOD-Test/QT+KW+KO/OOD-Test-QT+KW+KO-Ques.json", "r"))
	QTKWKO_qid = [k['question_id'] for k in QTKWKO_file]

	return QT_qid, KW_qid, KWP_qid, QTKW_qid, KO_qid, KOP_qid, QTKO_qid, KWKO_qid, QTKWKO_qid

def get_OOD_ans_pred(annotations, predictions, QT_qid, KW_qid, KWP_qid, QTKW_qid, KO_qid, KOP_qid, QTKO_qid, KWKO_qid, QTKWKO_qid):
	QT_annotations=[]
	KW_annotations=[]
	KWP_annotations=[]
	QTKW_annotations=[]
	KO_annotations=[]
	KOP_annotations=[]
	QTKO_annotations=[]
	KWKO_annotations=[]
	QTKWKO_annotations=[]
	QT_predictions=[]
	KW_predictions=[]
	KWP_predictions=[]
	QTKW_predictions=[]
	KO_predictions=[]
	KOP_predictions=[]
	QTKO_predictions=[]
	KWKO_predictions=[]
	QTKWKO_predictions=[]

	for x, y in zip(annotations, predictions):
		assert x['question_id'] == y['question_id']
		if x['question_id'] in QT_qid:
			QT_annotations.append(x)
			QT_predictions.append(y)
		if x['question_id'] in KW_qid:
			KW_annotations.append(x)
			KW_predictions.append(y)
		if x['question_id'] in KWP_qid:
			KWP_annotations.append(x)	
			KWP_predictions.append(y)
		if x['question_id'] in QTKW_qid:
			QTKW_annotations.append(x)	
			QTKW_predictions.append(y)	
		if x['question_id'] in KO_qid:
			KO_annotations.append(x)
			KO_predictions.append(y)
		if x['question_id'] in KOP_qid:
			KOP_annotations.append(x)
			KOP_predictions.append(y)
		if x['question_id'] in QTKO_qid:
			QTKO_annotations.append(x)
			QTKO_predictions.append(y)
		if x['question_id'] in KWKO_qid:
			KWKO_annotations.append(x)
			KWKO_predictions.append(y)
		if x['question_id'] in QTKWKO_qid:
			QTKWKO_annotations.append(x)
			QTKWKO_predictions.append(y)
			
	return 	(QT_annotations,KW_annotations,KWP_annotations,QTKW_annotations,KO_annotations,KOP_annotations,QTKO_annotations,KWKO_annotations,QTKWKO_annotations),(QT_predictions,KW_predictions,KWP_predictions,QTKW_predictions,KO_predictions,KOP_predictions,QTKO_predictions,KWKO_predictions,QTKWKO_predictions)

if __name__ == '__main__':

	args = parse_args()

	anno_path = osp.join(args.dataroot, '%s_target_count.pth'%(args.name))
	annotations = torch.load(anno_path)

	annotations = sorted(annotations, key=lambda x: x['question_id'])
	predictions = sorted(json.load(open(args.input)), key=lambda x: x['question_id'])
	iid_score=get_scores(annotations, predictions)

	QT_qid, KW_qid, KWP_qid, QTKW_qid, KO_qid, KOP_qid, QTKO_qid, KWKO_qid, QTKWKO_qid = get_qid()
	(QT_annotations,KW_annotations,KWP_annotations,QTKW_annotations,KO_annotations,KOP_annotations,QTKO_annotations,KWKO_annotations,QTKWKO_annotations),(QT_predictions,KW_predictions,KWP_predictions,QTKW_predictions,KO_predictions,KOP_predictions,QTKO_predictions,KWKO_predictions,QTKWKO_predictions)=get_OOD_ans_pred(annotations, predictions, QT_qid, KW_qid, KWP_qid, QTKW_qid, KO_qid, KOP_qid, QTKO_qid, KWKO_qid, QTKWKO_qid)



	QT_score=get_scores(QT_annotations, QT_predictions)
	
	KW_score=get_scores(KW_annotations, KW_predictions)

	KWP_score=get_scores(KWP_annotations, KWP_predictions)

	QTKW_score=get_scores(QTKW_annotations, QTKW_predictions)

	KO_score=get_scores(KO_annotations, KO_predictions)

	KOP_score=get_scores(KOP_annotations, KOP_predictions)
	
	QTKO_score=get_scores(QTKO_annotations, QTKO_predictions)

	KWKO_score=get_scores(KWKO_annotations, KWKO_predictions)

	QTKWKO_score=get_scores(QTKWKO_annotations, QTKWKO_predictions)

	print("iid_score", iid_score, "QT_score", QT_score, "KW_score", KW_score, "KWP_score", KWP_score, "QTKW_score", QTKW_score, "KO_score", KO_score, "KOP_score", KOP_score, "QTKO_score", QTKO_score, "KWKO_score", KWKO_score, "QTKWKO_score", QTKWKO_score)


	



