# VQA-VS (Language Prior Is Not the Only Shortcut: A Benchmark for Shortcut Learning in VQA)

Here is the implementation of our Findings of EMNLP-2022 [Language Prior Is Not the Only Shortcut: A Benchmark for Shortcut Learning in VQA](https://github.com/PhoebusSi/VQA-VS/)


## Motivation of our benchmark.
![image](https://github.com/PhoebusSi/VQA-VS/blob/main/figures/motivations.jpg)
Figure 1: (a) The acc improvement of LMH over its backbone model UpDn on nine OOD test sets. (The acronyms, like QT, are defined in Sec. 3.2 of paper) (b) Solutions possibly learnd by models.

As shown in Fig. 1(a), despite performing well on VQA-CP v2, the debi- asing method LMH (Clark et al., 2019), can only boost its backbone model UpDn on few certain OOD test sets while fails to generalize to other OOD sets. This shows VQA-CP v2 cannot identify whether the models rely on other types of short- cuts (e.g., correlations between visual objects and answers). Therefore, as shown in Fig. 1(b), more OOD test sets are needed to measure the reliance of the model on different types of shortcuts. As the performance on more OOD test sets is improved simultaneously, the more confidently can the model be deemed to have learned the intended solution.

## Examples from our dataset. 
![image](https://github.com/PhoebusSi/VQA-VS/blob/main/figures/examples.jpg)
Figure 2: Examples from our dataset. Each sample is labeled with nine shortcut-specific concepts.

## Data statistics.
![image](https://github.com/PhoebusSi/VQA-VS/blob/main/figures/data_statistics.jpg)
Table 1: Data statistics of VQA-VS (bold) and nine shortcuts.


## Relevance of Shortcuts & Overlaps Between OOD Test Sets.
![image](https://github.com/PhoebusSi/VQA-VS/blob/main/figures/relevance-overlaps.jpg)
Figure 3: (a) The Jaccard Similarity Coefficients be- tween all head splits of the training set. The higher the value, the closer the two types of shortcuts. (b) The co-incidence ratios between all OOD test sets. The square with coordinate (KO, QT) denotes that the proportion of the duplicate samples between KO and QT in the QT OOD test set.

Relevance of Shortcuts. The samples of head splits, which are frequent and dominating the model training (Goyal et al., 2017), are the main cause of the shortcuts in training data. Therefore, we use the relevance of two shortcuts’ head splits in training set to analyze the relevance of two short- cuts. As shown in Fig. 5(a), the Jaccard Simliarity Coefficient between QT and KO shortcuts is obvi- ously higher. A possible explanation is that there is a strong correlation between question types and key-object types. For example, the question type "who is" and key-object type "person" co-occurfrequently. Moreover, the KOP shortcut is closely relevant with KO because the shortcut concepts of KOP are involved with KO concepts. Consequently, QT is highly relevant with KOP. The relevance ex- tends to some of the other shortcuts in the same way, which can explain the light pink squares of Fig. 5(a). Differently, the coarse-grained QT and the fine-grained QT+KW have a low relevance even though the concepts of QT+KW include the QT concepts. This shows the necessity of introducing more fine-grained shortcuts which focus on a large number of concepts. 

Overlaps Between OOD Test Sets. Intuitively, if two OOD test sets share too many samples, there is no need to separately evaluate the model on the two OOD test sets. To rule out this possibility and valid the necessity of nine OOD test sets, we count the numbers of duplicate samples between all OOD test sets and compute corresponding coincidence rates. From Fig. 5(b), we find that the coincidence ratios between most OOD test sets are low. Al- though (KO, QT) has a high coincidence rate 0.79, the coincidence rate of (QT, KO) is much lower, 0.49, which shows the KO has a different emphasis compared with QT.
