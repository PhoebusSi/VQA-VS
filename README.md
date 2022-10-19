# VQA-VS (Language Prior Is Not the Only Shortcut: A Benchmark for Shortcut Learning in VQA)

Here is the data and implementation of our Findings of EMNLP-2022 [Language Prior Is Not the Only Shortcut: A Benchmark for Shortcut Learning in VQA](https://arxiv.org/abs/2210.04692).

The VQA-VS [**homepage**](https://phoebussi.github.io/VQA-VS-homepage/) is already accessible.

## Download the data.
### Approach 1: Google Drive
You can download the data from [**GoogleDrive**](https://drive.google.com/drive/folders/1i6xqke5X5GoQ8YGoNcs3rtMsDtgs4OLG?usp=sharing).

### Approach 2: Zip Compressed File
You can download the compressed files from the **data** folder according to its dir structure, and extract them separately.

### Approach 3: Contact me
You can contact me by email **siqingyi@iie.ac.cn**, and I will send the complete dataset to you.

## Data Preprocess.
### 1. Images of VQA-VS
Note that our proposed benchmark is re-organized from VQA v2, therefore, the images in the VQA-CP v1 and v2 datasets (both train and test) are from [training](http://images.cocodataset.org/zips/train2014.zip) and [validation](http://images.cocodataset.org/zips/val2014.zip) sets of the COCO dataset. 
Then you can map the images for the Training/Val/IID-Test/OOD-Test set according to their image-ids (annotated in the **-Ques.json** files).

For simplicity, same as the practice for VQA v2 or VQA-CP v2, you can also download the image features (extracted by FasterRCNN) by:
```
wget -P https://imagecaption.blob.core.windows.net/imagecaption/trainval_36.zip
(Alternate Link) wget -P https://storage.googleapis.com/up-down-attention/trainval_36.zip
unzip coco/trainval_36.zip -d image_features/
```
### 2. Preprocess the image features
```
python3 tsv2feature.py
```


## Reimplementations of the Baseline Framework.
### Preprocess the text data.
```
python3 preprocess_text.py
```
This step is neccessary if you use the framework we released.
### Training.
```
python3 main_LXM.py
```
The model which performs best on val dataset will be saved in the "saved_models" folder.

### Test and compute scores.
#### Test instructions
```
python3 test_LXM.py
```
The JSON file of your predictions on test set is saved in the "saved_models" folder.

Note that all OOD test sets are the subsets of the IID test set. Therefore, you can choose to predict the answers for the questions of each OOD test set seperately to get their test accuracy, or you can choose to predict the answers for the questions of the IID test set directly (**highly recommended**), and then collect the corresponding prediction results according to the question-id of each OOD test set to get their test accuracy. 

#### Compute scores
```
python3 compute_scores.py
```
This instruction can obtain the scores of IID test set and nine OOD test sets st the same time. You only need to pass in the predictions on IID test set. 
```
python3 new_compute_scores.py
```
We collected all the annotations needed for calculating the scores and packed them into *test_annotations.json* file. You can also get all the scores through this command.


## Metrics.
We use the common VQA evaluation metric: 
、、、
acc=min(#humans that provided that answer/3, 1),
、、、
i.e., an answer is deemed 100% accurate if at least 3 annotators provided that exact answer.
If you have the predictions for test set, you can compute scores as the function:
```
def cal_acc_multi(ground_truth, preds):
##ground_truth: [[a_1^1, a_1^2, ..., a_1^{10}], ..., [a_{64}^1, a_{64}^2, ..., a_{64}^{10}]] 
##preds: [p_1, p_2, ..., p_64] 
    all_num = len(ground_truth)
    acc_num = 0
    temp = []
    for i, answer_id in enumerate(ground_truth):
        pred = preds[i]
        cnt = 0
        for aid in answer_id:
            if pred == aid:
                cnt += 1
        if cnt ==1:
            acc_num += 1/3
        elif cnt == 2:
            acc_num += 2/3
        elif cnt > 2:
            acc_num += 1
        return acc_num/all_num
```

## Motivation of our benchmark.
![image](https://github.com/PhoebusSi/VQA-VS/blob/main/figures/motivations.jpg)
Figure 1: (a) The acc improvement of LMH over its backbone model UpDn on nine OOD test sets. (The acronyms, like QT, are defined in Sec. 3.2 of paper) (b) Solutions possibly learnd by models.

As shown in Fig. 1(a), despite performing well on VQA-CP v2, the debi- asing method LMH (Clark et al., 2019), can only boost its backbone model UpDn on few certain OOD test sets while fails to generalize to other OOD sets. This shows VQA-CP v2 cannot identify whether the models rely on other types of short- cuts (e.g., correlations between visual objects and answers). Therefore, as shown in Fig. 1(b), more OOD test sets are needed to measure the reliance of the model on different types of shortcuts. As the performance on more OOD test sets is improved simultaneously, the more confidently can the model be deemed to have learned the intended solution.

## Examples from our dataset. 
![image](https://github.com/PhoebusSi/VQA-VS/blob/main/figures/examples.jpg)
Figure 2: Examples from our dataset. Each sample is labeled with nine shortcut-specific concepts.

A concept can be viewed as an instance of the corresponding shortcut and represents the most salient information that is likely to be associated with the answer. For example, given a sample with the question "What color is the banana?", "what color" is the concept selected for the QT shortcut, and "banana" is the concept selected for the KW shortcut. Fig. 2 shows examples with nine shortcut-specific concepts 

## Data statistics.
![image](https://github.com/PhoebusSi/VQA-VS/blob/main/figures/data-statistics.jpg)
Table 1: Data statistics of VQA-VS (bold) and nine shortcuts.

Tab. 1 shows the data statistics of VQA-VS, and the group and sample statistics for each shortcut. The total numbers of groups vary significantly among different shortcuts (65 ~183683).


## Relevance of Shortcuts & Overlaps Between OOD Test Sets.
![image](https://github.com/PhoebusSi/VQA-VS/blob/main/figures/relevance-overlaps.jpg)
Figure 3: (a) The Jaccard Similarity Coefficients be- tween all head splits of the training set. The higher the value, the closer the two types of shortcuts. (b) The co-incidence ratios between all OOD test sets. The square with coordinate (KO, QT) denotes that the proportion of the duplicate samples between KO and QT in the QT OOD test set.

Relevance of Shortcuts. The samples of head splits, which are frequent and dominating the model training (Goyal et al., 2017), are the main cause of the shortcuts in training data. Therefore, we use the relevance of two shortcuts’ head splits in training set to analyze the relevance of two short- cuts. As shown in Fig. 3(a), the Jaccard Simliarity Coefficient between QT and KO shortcuts is obvi- ously higher. A possible explanation is that there is a strong correlation between question types and key-object types. For example, the question type "who is" and key-object type "person" co-occurfrequently. Moreover, the KOP shortcut is closely relevant with KO because the shortcut concepts of KOP are involved with KO concepts. Consequently, QT is highly relevant with KOP. The relevance ex- tends to some of the other shortcuts in the same way, which can explain the light pink squares of Fig. 5(a). Differently, the coarse-grained QT and the fine-grained QT+KW have a low relevance even though the concepts of QT+KW include the QT concepts. This shows the necessity of introducing more fine-grained shortcuts which focus on a large number of concepts. 

Overlaps Between OOD Test Sets. Intuitively, if two OOD test sets share too many samples, there is no need to separately evaluate the model on the two OOD test sets. To rule out this possibility and valid the necessity of nine OOD test sets, we count the numbers of duplicate samples between all OOD test sets and compute corresponding coincidence rates. From Fig. 3(b), we find that the coincidence ratios between most OOD test sets are low. Although (KO, QT) has a high coincidence rate 0.79, the coincidence rate of (QT, KO) is much lower, 0.49, which shows the KO has a different emphasis compared with QT.

## Comparison of our benchmark and VQA-CP v2.
![image](https://github.com/PhoebusSi/VQA-VS/blob/main/figures/comparison.jpg)
Table 2: Comparison of our benchmark and VQA-CP v2. The results are computed over four seeds.


## Performance of SoTA debiasing methods.
![image](https://github.com/PhoebusSi/VQA-VS/blob/main/figures/sotas.jpg)
Table 3: Performance of SoTA debiasing methods.


## Reference
If you found this code is useful, please cite the following paper:
```
@article{Si2022LanguagePI,
  title={Language Prior Is Not the Only Shortcut: A Benchmark for Shortcut Learning in VQA},
  author={Qingyi Si and Fandong Meng and Mingyu Zheng and Zheng Lin and Yuanxin Liu and Peng Fu and Yanan Cao and Weiping Wang and Jie Zhou},
  journal={ArXiv},
  year={2022},
  volume={abs/2210.04692}
}
```
