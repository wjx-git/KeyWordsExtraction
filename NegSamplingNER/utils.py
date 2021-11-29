from tqdm import tqdm
import time
import pickle as pkl
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from pytorch_pretrained_bert import BertTokenizer
from torch.utils.tensorboard import SummaryWriter

from NegSamplingNER.misc import extract_json_data
from NegSamplingNER.misc import iob_tagging, f1_score
from ltp import LTP

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
ltp = LTP()


class UnitAlphabet(object):

    CLS_SIGN, SEP_SIGN = "[CLS]", "[SEP]"
    PAD_SIGN, UNK_SIGN = "[PAD]", "[UNK]"

    def __init__(self, source_path):
        self._tokenizer = BertTokenizer.from_pretrained(source_path, do_lower_case=True)

    def tokenize(self, item):
        return self._tokenizer.tokenize(item)

    def index(self, items):
        return self._tokenizer.convert_tokens_to_ids(items)


class LabelAlphabet(object):

    def __init__(self):
        super(LabelAlphabet, self).__init__()

        self._idx_to_item = []
        self._item_to_idx = {}

    def add(self, item):
        if item not in self._item_to_idx:
            self._item_to_idx[item] = len(self._idx_to_item)
            self._idx_to_item.append(item)

    def get(self, idx):
        return self._idx_to_item[idx]

    def index(self, item):
        return self._item_to_idx[item]

    def __str__(self):
        return str(self._item_to_idx)

    def __len__(self):
        return len(self._idx_to_item)


class LexicalAlphabet(object):

    def __init__(self, vocab_file):
        super(LexicalAlphabet, self).__init__()

        self._idx_to_item, self._item_to_idx = self._load_vocab(vocab_file)

    def add(self, item):
        if item not in self._item_to_idx:
            self._item_to_idx[item] = len(self._idx_to_item)
            self._idx_to_item.append(item)

    def _load_vocab(self, file):
        """
        加载词表，词表格式：id: token,  转换成 token: id
        :param file:
        :return:
        """
        with open(file, 'rb') as f:
            vocab = pkl.load(f)
        return vocab.keys(), dict(zip(vocab.values(), vocab.keys()))

    def get(self, idx):
        return self._idx_to_item[idx]

    def index(self, item, unk_id=None):
        return self._item_to_idx.get(item, unk_id)

    def __str__(self):
        return str(self._item_to_idx)

    def __len__(self):
        return len(self._idx_to_item)


def corpus_to_iterator(file_path, batch_size, if_shuffle, label_vocab=None):
    material = extract_json_data(file_path)
    instances = [(eval(e["sentence"]), eval(e["labeled entities"])) for e in material]
    instances = instances[:80]

    if label_vocab is not None:
        label_vocab.add("O")
        for _, u in instances:
            for _, _, l in u:
                label_vocab.add(l)

    class _DataSet(Dataset):

        def __init__(self, elements):
            self._elements = elements

        def __getitem__(self, item):
            return self._elements[item]

        def __len__(self):
            return len(self._elements)

    def distribute(elements):
        sentences, entities, words = [], [], []
        for s, e in elements:
            sentences.append(s)
            entities.append(e)
            seg, hidden = ltp.seg([''.join(s)])
            word_start_pos = [0]
            for w in seg[0]:
                word_start_pos.append(word_start_pos[-1] + len(w))
            words.append(word_start_pos)
        return sentences, entities, words

    wrap_data = _DataSet(instances)
    return DataLoader(wrap_data, batch_size, if_shuffle, collate_fn=distribute)


writer = SummaryWriter('logs')


class Procedure(object):

    @staticmethod
    def train(model, dataset, optimizer):
        model.train()
        time_start, total_penalties = time.time(), 0.0
        global_step = 0

        for sentences, segments, words in tqdm(dataset, ncols=50):
            loss = model.estimate(sentences, segments, words)
            total_penalties += loss.cpu().item()
            global_step += 1
            writer.add_scalar('train_loss', loss, global_step=global_step)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        time_con = time.time() - time_start
        return total_penalties, time_con

    @staticmethod
    def dev(model, dataset, eval_path, epoch_i):
        model.eval()
        time_start, total_penalties = time.time(), 0.0
        global_step = 0
        seqs, outputs, oracles = [], [], []

        for sentences, segments, words in tqdm(dataset, ncols=50):
            with torch.no_grad():
                loss, predictions = model.dev_estimate(sentences, segments, words)
            global_step += 1
            writer.add_scalar('dev_loss', loss, global_step=global_step)

            seqs.extend(sentences)
            outputs.extend([iob_tagging(e, len(u)) for e, u in zip(predictions, sentences)])
            oracles.extend([iob_tagging(e, len(u)) for e, u in zip(segments, sentences)])
        out_f1 = f1_score(seqs, outputs, oracles, eval_path, epoch_i)
        time_con = time.time() - time_start
        return out_f1, time_con

    @staticmethod
    def test(model, dataset, eval_path=None, epoch_i=None):
        model.eval()
        # time_start = time.time()
        seqs, outputs, oracles = [], [], []
        # results = {}

        for sentences, segments, words in tqdm(dataset, ncols=50):
            with torch.no_grad():
                predictions = model.inference(sentences, words)

            # seqs.extend(sentences)
            # outputs.extend([iob_tagging(e, len(u)) for e, u in zip(predictions, sentences)])
            # oracles.extend([iob_tagging(e, len(u)) for e, u in zip(segments, sentences)])

            for sent, pred in zip(sentences, predictions):
                outputs.append({'sentence': ''.join(sent), 'label': pred})

        # out_f1 = f1_score(seqs, outputs, oracles, eval_path, epoch_i)
        # return out_f1, time.time() - time_start
        import json
        with open('dataset/pred_roberta.json', 'w', encoding='utf-8') as f:
            json.dump(outputs, f, ensure_ascii=False, indent=2)


def build_char_embedding_and_vocab(pre_train_file, char_embedding_file, word_file, dim):
    """
    从预训练文件提取字向量和字表。
    预训练文件每行数据格式：字 + 向量
    词向量文件：保存从预训练文件中提取的词向量，以字典形式存储，key为词向量所在行索引，value为词向量
    词表文件：保存从预训练文件中提取的词，以字典形式存储，key为词所在行索引，value为词

    :param pre_train_file: 预训练文件路径
    :param char_embedding_file: 词向量文件路径
    :param word_file: 词表文件路径
    """
    embeddings = []
    vocabs = {}
    digit_emb = []
    n = 1
    with open(pre_train_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f.readlines()):
            line = line.strip().split(' ')
            if line[0].isdigit():
                digit_emb.append(list(map(float, line[1:])))
            else:
                vocabs[n] = line[0]
                n += 1
                embeddings.append(list(map(float, line[1:])))
        vocabs[n] = '<number>'
        digit_emb = np.array(digit_emb)
        digit_emb = digit_emb.sum(axis=0) / len(digit_emb)
        embeddings.append(list(digit_emb))

    # 增加补充词
    vocabs[0] = '<PAD>'
    vocabs[n+1] = '<UNK>'
    pad_e = np.zeros((1, dim))
    unk_e = np.random.rand(1, dim)
    embeddings = np.concatenate([pad_e, embeddings, unk_e], axis=0)
    assert len(vocabs) == len(embeddings)

    np.savez_compressed(char_embedding_file, embeddings=embeddings)
    with open(word_file, 'wb') as f:
        pkl.dump(vocabs, f)
    print('word counts:', len(vocabs))


if __name__ == '__main__':
    pw = r'D:\ProgramData\WordsEmbeddingData\gloveEN/glove.6B.300d.txt'
    ce = 'resource/glove/glove_6B_300d.npz'
    wd = 'resource/glove/word_300d.pkl'
    build_char_embedding_and_vocab(pw, ce, wd, 300)
