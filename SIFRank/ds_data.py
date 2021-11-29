"""
@Project ：SIFRank
@File ：ds_data.py
@IDE ：PyCharm
"""
import csv
import json
import re
import random

from tqdm import tqdm

from SIFRank.unsupervised_extract import extract


def read_data(file):
    question = set()
    pat = re.compile('[*#]')
    with open(file, 'r', encoding='utf-8') as f:
        for line in csv.reader(f):
            if not re.search(pat, line[0]):
                question.add(line[0])
    return question


def save_json(file, data):
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def label_sample(samples):
    counts = 0
    labeled_samples = []
    for sample in tqdm(samples):
        try:
            label = extract(sample)
            entity = []
            if label:
                for (word, score) in label[:1]:  # top 1
                    if score >= 0.95:
                        start = sample.find(word)
                        if start != -1:
                            entity.append((start, start+len(word)-1, 'KW'))
                counts += 1
            sample = [w for w in sample]
            labeled_samples.append({'sentence': str(sample), 'labeled entities': str(entity)})
        except:
            pass
        if counts >= 30000:
            break
    print(counts)
    return labeled_samples


def train_dev_test(samples):
    random.shuffle(samples)
    totals = len(samples)
    train = samples[:int(totals * 0.7)]
    dev = samples[int(totals * 0.7): -200]
    test = samples[-200:]

    save_json('SIFRank/data/train_roberta_top1.json', train)
    save_json('SIFRank/data/dev_roberta_top1.json', dev)
    save_json('SIFRank/data/test_roberta_top1.json', test)


def matrix(preds, labels):
    tp, fp, fn = 0, 0, 0
    for pred, label in zip(preds, labels):
        for p in pred:
            if p in label:
                tp += 1
            else:
                fp += 1
        for l in label:
            if l not in pred:
                fn += 1
    recall = tp / (tp + fn)
    prec = tp / (tp + fp)
    f1 = 2 * recall * prec / (recall + prec)
    print('P: {}, R:{}, F1:{}'.format(prec, recall, f1))


def distance_label(infile):
    """生成远程标注数据"""
    liantong = read_data(infile)
    ds_data = label_sample(liantong)
    train_dev_test(ds_data)


if __name__ == '__main__':
    distance_label('SIFRank/data/liantongzhidao_filter.csv')

    # with open('./data/test_roberta_top1.json', 'r', encoding='utf-8') as f:
    #     samples = json.load(f)
    # instances = [(eval(e["sentence"]), eval(e["labeled entities"])) for e in samples]
    # sentence = []
    # for ins in instances:
    #     print([w + '-' + str(i) for i, w in enumerate(ins[0])])

    # 计算测试结果
    # with open('data/test.json', 'r', encoding='utf-8') as f:
    #     samples = json.load(f)
    # instances = [(eval(e["sentence"]), eval(e["labeled entities"])) for e in samples]
    # sentences = [''.join(eval(e["sentence"])) for e in samples]
    # labels = [eval(e["labeled entities"]) for e in samples]
    # labeled_samples = label_sample(sentences)
    # preds = [eval(ls.get('labeled entities')) for ls in labeled_samples]
    # matrix(preds, labels)
