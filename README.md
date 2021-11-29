# 中文短文本关键词抽取

关键词抽取有无监督方法和有监督方法，显然有监督方法效果更好，但是没有标注语料的
场景下无法训练监督学习模型。本文结合无监督方法和有监督方法完成无标注语料场景下关键词抽取。

我们采用的中文短文本关键词抽取方案有两个步骤：
1. 首先使用无监督学习方法[SIFRank](https://github.com/sunyilgdx/SIFRank_zh) 标注数据，
2. 然后训练命名实体识别模型[NegSamplingNER](https://github.com/LeePleased/NegSampling-NER) 。

SIFRank用于无监督标注语料，为了保障模型标注的词尽可能是正确的，每个句子只标注排序第一的词，
但这样会造成漏标情况。漏标数据对于训练监督学习模型带来了大量噪声，
为了减少噪声影响，我们选择训练NegSamplingNER命名实体识别模型，
该方法专为用于远程监督数据而提出，其采用的片段采样方法有效减少漏标数据影响。

## 用法
1. 使用SIFRank方法标注数据

    1.1在SIFRank文件夹下调用ds_data.py中的distance_label函数，
       或者自行调用unsupervised_extract.py中的extract函数；
    
    1.2 将生成的文件放到 NegSamplingNER/data目录下；

2. 训练NegSamplingNER模型

    2.1 在NegSamplingNER文件夹下运行main.py文件训练模型。
