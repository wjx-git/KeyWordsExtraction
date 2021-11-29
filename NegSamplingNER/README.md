# Negative Sampling for NER

*Unlabeled entity problem* is prevalent in many NER scenarios (e.g., weakly supervised NER). 
Our [paper](https://openreview.net/forum?id=5jRVa89sZk) in ICLR-2021 proposes using negative sampling for solving this important issue.
This repo. contains the implementation of our approach.

Note that this is not an officially supported Tencent product.

注：本文修改了预测时片段切分方法，原文是按照token切分片段，这会导致一个词被切分到不同
片段，例如“looking"分词后为”look, ##ing"，“##ing”可能被切分为单独片段。
在中文中一个词也可能被切分为不同片段，例如“广州”被切分为两个字，我们按照词粒度切分片段，
目的是减少非常规片段作为负样本，增大模型学习难度。

## Preparation

Two steps. Firstly, reformulate the NER data and move it into a new folder named "dataset". 
The folder contains {train, dev, test}.json. 
Each JSON file is a list of dicts. See the following case:
```
[ 
 {
  "sentence": "['Somerset', '83', 'and', '174', '(', 'P.', 'Simmons', '4-38', ')', ',', 'Leicestershire', '296', '.']",
  "labeled entities": "[(0, 0, 'ORG'), (5, 6, 'PER'), (10, 10, 'ORG')]",
 },
 {
  "sentence": "['Leicestershire', '22', 'points', ',', 'Somerset', '4', '.']",
  "labeled entities": "[(0, 0, 'ORG'), (4, 4, 'ORG')]",
 }
]
```

Secondly, pretrained LM (i.e., [BERT](https://www.aclweb.org/anthology/N19-1423/)) and [eval. script](https://www.clips.uantwerpen.be/conll2000/chunking/conlleval.txt). 
Create a dir. named "resource" and arrange them as
- resource
    - bert-base-cased
        - model.pt
        - vocab.txt
    - conlleval.pl

Note that the files in BERT.tar.gz need to be renamed as above.

## Training and Test
```
CUDA_VISIBLE_DEVICES=0 python main.py -dd dataset -cd save -rd resource
```

## Citation
```
@inproceedings{li2021empirical,
    title={Empirical Analysis of Unlabeled Entity Problem in Named Entity Recognition},
    author={Yangming Li and lemao liu and Shuming Shi},
    booktitle={International Conference on Learning Representations},
    year={2021},
    url={https://openreview.net/forum?id=5jRVa89sZk}
}
```
