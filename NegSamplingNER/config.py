from NegSamplingNER.misc import extract_json_data
import json
import random
from collections import defaultdict


def check_sample():
    file_path = 'dataset/test2.json'
    material = extract_json_data(file_path)
    instances = [(eval(e["sentence"]), eval(e["labeled entities"])) for e in material]

    samples = []
    for sent, entities in instances:
        sentence = ''.join(sent)
        labels = []
        for entity in entities:
            labels.append(''.join(sent[entity[0]: entity[1] + 1]) + ' ' + entity[2])
        samples.append({'sentence': sentence, 'labeled entities': labels})

    random.seed(10)
    random.shuffle(samples)
    samples1 = samples[:100]
    with open('dataset/check1.json', 'w', encoding='utf-8') as f:
        json.dump(samples1, f, ensure_ascii=False, indent=4)

    samples2 = samples[100:200]
    with open('dataset/check2.json', 'w', encoding='utf-8') as f:
        json.dump(samples2, f, ensure_ascii=False, indent=4)


def count_sample():
    file_path = 'dataset/check2.json'
    with open(file_path, 'r', encoding='utf-8') as f:
        samples = json.load(f)

    labels = defaultdict(int)
    for sample in samples:
        entities = sample.get('labeled entities', [])
        for entity in entities:
            _, entity_type = entity.split()
            labels[entity_type] += 1
    print(labels)


if __name__ == '__main__':
    count_sample()


