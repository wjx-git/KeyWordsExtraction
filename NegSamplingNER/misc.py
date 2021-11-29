import time
import json
import codecs
import os
import numpy as np
import random

import torch


def iterative_support(func, query):
    if isinstance(query, (list, tuple, set)):
        return [iterative_support(func, i) for i in query]
    return func(query)


def fix_random_seed(state_val):
    random.seed(state_val)
    np.random.seed(state_val)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(state_val)
        torch.cuda.manual_seed_all(state_val)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    torch.manual_seed(state_val)
    torch.random.manual_seed(state_val)


def flat_list(h_list):
    e_list = []

    for item in h_list:
        if isinstance(item, list):
            e_list.extend(flat_list(item))
        else:
            e_list.append(item)
    return e_list


def f1_score(sent_list, pred_list, gold_list, script_path, epoch_i):
    fn_out = 'resource/results.txt'
    # if os.path.isfile(fn_out):
    #     os.remove(fn_out)
    # if not os.path.exists('resource/results'):
    #     os.mkdir('resource/results')

    text_file = open(fn_out, mode='w')
    for i, words in enumerate(sent_list):
        tags_1 = gold_list[i]
        tags_2 = pred_list[i]
        for j, word in enumerate(words):
            tag_1 = tags_1[j]
            tag_2 = tags_2[j]
            text_file.write('%s %s %s\n' % (word, tag_1, tag_2))
        text_file.write('\n')
    text_file.close()

    # cmd = 'perl %s < %s' % (script_path, fn_out)  # linux系统上用这行
    cmd = '%s < %s' % (script_path, fn_out)  # window系统
    msg = '\nStandard CoNNL perl script (author: Erik Tjong Kim Sang <erikt@uia.ua.ac.be>, version: 2004-01-26):\n'
    msg += ''.join(os.popen(cmd).readlines())
    time.sleep(1.0)
    # if fn_out.startswith('resource/eval_') and os.path.exists(fn_out):
    #     os.remove(fn_out)
    with open('resource/res_epoch_{}.txt'.format(epoch_i), 'w', encoding='utf-8') as f:
        f.write(msg)
    return float(msg.split('\n')[3].split(':')[-1].strip())


def iob_tagging(entities, s_len):
    tags = ["O"] * s_len

    for el, er, et in entities:
        for i in range(el, er + 1):
            if i == el:
                tags[i] = "B-" + et
            else:
                tags[i] = "I-" + et
    return tags


def conflict_judge(line_x, line_y):
    if line_x[0] == line_y[0]:
        return True
    if line_x[0] < line_y[0]:
        if line_x[1] >= line_y[0]:
            return True
    if line_x[0] > line_y[0]:
        if line_x[0] <= line_y[1]:
            return True
    return False


def extract_json_data(file_path):
    with codecs.open(file_path, "r", "utf-8") as fr:
        dataset = json.load(fr)
    return dataset


def reformat_dataset(file_path):
    """
    生成训练格式数据
    :param file_path:
    :return:
    """
    samples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        data = {}
        sentence, label = [], []
        for line in f.readlines():
            if line == '\n' and len(sentence) > 0:
                data['sentence'] = str(sentence)
                labeled_entities = []
                begin, end = 0, 0
                begin_flag = False
                entity_type = ''
                for i, tag in enumerate(label):
                    if tag.startswith('B'):
                        begin = i
                        entity_type = tag[2:]
                        begin_flag = True
                    elif tag == 'O' and begin_flag:
                        end = i - 1
                        labeled_entities.append((begin, end, entity_type))
                        begin_flag = False
                if begin_flag:
                    labeled_entities.append((begin, i, entity_type))

                data['labeled entities'] = str(labeled_entities)
                samples.append(data)
                data = {}
                sentence, label = [], []
            else:
                item = line.split()
                sentence.append(item[0])
                label.append(item[1])
    return samples


def store_json(file_path, data, mode='w'):
    """
    保存文件，格式json
    :param file_path: 路径
    :param data: list，[{1: 'a'}, {2: 'b'}, {3: 'c'}]
    :param mode:
    :return:
    """
    with open(file_path, mode, encoding='utf-8') as f:
        f.write(json.dumps(data, indent=4, ensure_ascii=False))

#
# if __name__ == '__main__':
#     # samples = reformat_dataset('dataset/train.txt')
#     # store_json('dataset/train.json', samples)
#     from pytorch_pretrained_bert import BertModel, BertTokenizer
#     import numpy as np
#     import torch
#
#     # 加载bert的分词器
#     tokenizer = BertTokenizer.from_pretrained('resource/bert/vocab.txt')
#     # 加载bert模型，这个路径文件夹下有bert_config.json配置文件和model.bin模型权重文件
#     bert = BertModel.from_pretrained('resource/bert/')
#
#     s = "I'm not sure, this can work, lol -.-"
#
#     tokens = tokenizer.tokenize(s)
#     print("\\".join(tokens))
#
#     ids = torch.tensor([tokenizer.convert_tokens_to_ids(tokens)])
#     print(ids.shape)
#     # torch.Size([1, 15])
#
#     result = bert(ids, output_all_encoded_layers=True)
#     print(result)
#
#
#     def reformat_dataset():
#         """
#         生成训练格式数据
#         :param file_path:
#         :return:
#         """
#         samples = []
#         file_path = r'D:\ProgramData\NERdata\datasets_ner\nerdata_30W\all_test.nerdata'
#         with open(file_path, 'r', encoding='utf-8') as f:
#             data = {}
#             sentence, label = [], []
#             for line in f.readlines():
#                 if line == '\n' and len(sentence) > 0:
#                     data['sentence'] = str(sentence)
#                     labeled_entities = []
#                     begin, end = 0, 0
#                     begin_flag = False
#                     entity_type = ''
#                     for i, tag in enumerate(label):
#                         if tag.startswith('B'):
#                             begin = i
#                             entity_type = tag[2:]
#                             begin_flag = True
#                         elif tag.startswith('E') and begin_flag:
#                             end = i
#                             labeled_entities.append((begin, end, entity_type))
#                             begin_flag = False
#                         elif tag.startswith('S'):
#                             begin = i
#                             end = i
#                             entity_type = tag[2:]
#                             labeled_entities.append((begin, end, entity_type))
#                     if begin_flag:
#                         labeled_entities.append((begin, i, entity_type))
#
#                     data['labeled entities'] = str(labeled_entities)
#                     samples.append(data)
#                     data = {}
#                     sentence, label = [], []
#                 else:
#                     item = line.split()
#                     sentence.append(item[0])
#                     label.append(item[1])
#         return samples
#
#
#     samples = trans.reformat_dataset()
#     output_file = 'nerdata_30W/test.json'
#     with open(output_file, 'w', encoding='utf-8') as f:
#         json.dump(samples, f, indent=4, ensure_ascii=False)
