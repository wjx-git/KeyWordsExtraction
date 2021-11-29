import numpy as np

import torch
from torch import nn
from pytorch_pretrained_bert import BertModel

from NegSamplingNER.misc import flat_list
from NegSamplingNER.misc import iterative_support, conflict_judge
from NegSamplingNER.utils import UnitAlphabet, LabelAlphabet


class PhraseClassifier(nn.Module):

    def __init__(self,
                 lexical_vocab: UnitAlphabet,
                 label_vocab: LabelAlphabet,
                 hidden_dim: int,
                 dropout_rate: float,
                 neg_rate: float,
                 bert_path: str):
        super(PhraseClassifier, self).__init__()

        self._lexical_vocab = lexical_vocab
        self._label_vocab = label_vocab
        self._neg_rate = neg_rate

        self._encoder = BERT(bert_path)
        self._classifier = MLP(self._encoder.dimension * 4, hidden_dim, len(label_vocab), dropout_rate)
        self._criterion = nn.NLLLoss()

    def forward(self, var_h, **kwargs):
        con_repr = self._encoder(var_h, kwargs["mask_mat"], kwargs["starts"])

        batch_size, token_num, hidden_dim = con_repr.size()
        ext_row = con_repr.unsqueeze(2).expand(batch_size, token_num, token_num, hidden_dim)
        ext_column = con_repr.unsqueeze(1).expand_as(ext_row)
        table = torch.cat([ext_row, ext_column, ext_row - ext_column, ext_row * ext_column], dim=-1)
        return self._classifier(table)

    def _pre_process_input(self, utterances):
        lengths = [len(s) for s in utterances]
        max_len = max(lengths)
        pieces = iterative_support(self._lexical_vocab.tokenize, utterances)
        units, positions = [], []

        for tokens in pieces:
            units.append(flat_list(tokens))
            cum_list = np.cumsum([len(p) for p in tokens]).tolist()
            positions.append([0] + cum_list[:-1])

        sizes = [len(u) for u in units]
        max_size = max(sizes)
        cls_sign = self._lexical_vocab.CLS_SIGN
        sep_sign = self._lexical_vocab.SEP_SIGN
        pad_sign = self._lexical_vocab.PAD_SIGN
        pad_unit = [[cls_sign] + s + [sep_sign] + [pad_sign] * (max_size - len(s)) for s in units]
        starts = [[ln + 1 for ln in u] + [max_size + 1] * (max_len - len(u)) for u in positions]

        # var_unit = torch.LongTensor([self._lexical_vocab.index(u) for u in pad_unit])
        # attn_mask = torch.LongTensor([[1] * (lg + 2) + [0] * (max_size - lg) for lg in sizes])
        # var_start = torch.LongTensor(starts)

        var_unit = torch.tensor([self._lexical_vocab.index(u) for u in pad_unit], dtype=torch.long)
        attn_mask = torch.tensor([[1] * (lg + 2) + [0] * (max_size - lg) for lg in sizes], dtype=torch.long)
        var_start = torch.tensor(starts, dtype=torch.long)

        if torch.cuda.is_available():
            var_unit = var_unit.cuda()
            attn_mask = attn_mask.cuda()
            var_start = var_start.cuda()
        return var_unit, attn_mask, var_start, lengths

    def _pre_process_output(self, entities, lengths, words):
        positions, labels = [], []
        batch_size = len(entities)

        # positive samples set
        for utt_i in range(batch_size):
            for segment in entities[utt_i]:
                positions.append((utt_i, segment[0], segment[1]))
                labels.append(segment[2])

        # sampling negative samples set
        for utt_i in range(batch_size):
            cur_sample_words = words[utt_i]
            reject_set = [(e[0], e[1]) for e in entities[utt_i]]
            s_len = len(cur_sample_words)
            # pos_num = len(entities[utt_i]) if len(entities[utt_i]) else 1
            neg_num = int(s_len * self._neg_rate) + 1
            # neg_num = pos_num * self._neg_rate

            # candies = flat_list([[(i, j) for j in range(i, s_len) if (i, j) not in reject_set] for i in range(s_len)])
            candies = []
            for i in range(s_len-1):
                for j in range(i+1, s_len):
                    if (cur_sample_words[i], cur_sample_words[j]-1) not in reject_set:
                        candies.append((cur_sample_words[i], cur_sample_words[j]-1))
            candies = flat_list(candies)
            # candies = flat_list([[(words[i], words[j]-1) for j in range(i, s_len) if (words[i], words[j]-1) not in reject_set] for i in range(s_len)])
            if len(candies) > 0:
                sample_num = neg_num if neg_num <= len(candies) else len(candies)
                assert sample_num > 0

                np.random.shuffle(candies)
                for i, j in candies[:sample_num]:
                    positions.append((utt_i, i, j))
                    labels.append("O")

        # var_lbl = torch.LongTensor(iterative_support(self._label_vocab.index, labels))
        var_lbl = torch.tensor(iterative_support(self._label_vocab.index, labels), dtype=torch.long)
        if torch.cuda.is_available():
            var_lbl = var_lbl.cuda()
        return positions, var_lbl

    def estimate(self, sentences, segments, words):
        var_sent, attn_mask, start_mat, lengths = self._pre_process_input(sentences)
        score_t = self(var_sent, mask_mat=attn_mask, starts=start_mat)

        positions, targets = self._pre_process_output(segments, lengths, words)
        flat_s = torch.cat([score_t[[i], j, k] for i, j, k in positions], dim=0)
        return self._criterion(torch.log_softmax(flat_s, dim=-1), targets)

    def dev_estimate(self, sentences, segments, words):
        var_sent, attn_mask, starts, lengths = self._pre_process_input(sentences)
        log_items = self(var_sent, mask_mat=attn_mask, starts=starts)

        positions, targets = self._pre_process_output(segments, lengths, words)
        flat_s = torch.cat([log_items[[i], j, k] for i, j, k in positions], dim=0)

        loss = self._criterion(torch.log_softmax(flat_s, dim=-1), targets)

        score_t = torch.log_softmax(log_items, dim=-1)
        val_table, idx_table = torch.max(score_t, dim=-1)

        listing_it = idx_table.cpu().numpy().tolist()
        listing_vt = val_table.cpu().numpy().tolist()
        label_table = iterative_support(self._label_vocab.get, listing_it)

        candidates = []
        k = 0
        for l_mat, v_mat, sent_l in zip(label_table, listing_vt, lengths):
            candidates.append([])
            words_start_pos = words[k]
            sent_l = len(words_start_pos)
            for i in range(sent_l-1):
                p1 = words_start_pos[i]
                for j in range(i+1, sent_l):
                    p2 = words_start_pos[j] - 1
                    if l_mat[p1][p2] != "O":
                        candidates[-1].append((p1, p2, l_mat[p1][p2], v_mat[p1][p2]))
            k += 1

        entities = []
        for segments in candidates:
            ordered_seg = sorted(segments, key=lambda e: -e[-1])
            filter_list = []
            for elem in ordered_seg:
                flag = False
                current = (elem[0], elem[1])
                for prior in filter_list:
                    flag = conflict_judge(current, (prior[0], prior[1]))
                    if flag:
                        break
                if not flag:
                    filter_list.append((elem[0], elem[1], elem[2]))
            entities.append(sorted(filter_list, key=lambda e: e[0]))
        return loss, entities

    def inference(self, sentences, words):
        var_sent, attn_mask, starts, lengths = self._pre_process_input(sentences)
        log_items = self(var_sent, mask_mat=attn_mask, starts=starts)

        score_t = torch.log_softmax(log_items, dim=-1)
        val_table, idx_table = torch.max(score_t, dim=-1)

        listing_it = idx_table.cpu().numpy().tolist()
        listing_vt = val_table.cpu().numpy().tolist()
        label_table = iterative_support(self._label_vocab.get, listing_it)

        candidates = []
        k = 0
        for l_mat, v_mat, sent_l in zip(label_table, listing_vt, lengths):
            words_start_pos = words[k]
            sent_l = len(words_start_pos)
            candidates.append([])
            for i in range(sent_l-1):
                p1 = words_start_pos[i]
                for j in range(i+1, sent_l):
                    p2 = words_start_pos[j]-1
                    if l_mat[p1][p2] != "O":
                        candidates[-1].append((p1, p2, l_mat[p1][p2], v_mat[p1][p2]))
            k += 1

        entities = []
        for segments in candidates:
            ordered_seg = sorted(segments, key=lambda e: -e[-1])
            filter_list = []
            for elem in ordered_seg:
                # flag = False
                # current = (elem[0], elem[1])
                # for prior in filter_list:
                #     flag = conflict_judge(current, (prior[0], prior[1]))
                #     if flag:
                #         break
                # if not flag:
                #     filter_list.append((elem[0], elem[1], elem[2]))
                filter_list.append((elem[0], elem[1], elem[2]))
            entities.append(sorted(filter_list, key=lambda e: e[0]))
        return entities


class BERT(nn.Module):

    def __init__(self, source_path):
        super(BERT, self).__init__()
        self._repr_model = BertModel.from_pretrained(source_path)

    @property
    def dimension(self):
        return 768

    def forward(self, var_h, attn_mask, starts):
        all_hidden, _ = self._repr_model(var_h, attention_mask=attn_mask, output_all_encoded_layers=False)

        batch_size, _, hidden_dim = all_hidden.size()
        _, unit_num = starts.size()
        positions = starts.unsqueeze(-1).expand(batch_size, unit_num, hidden_dim)
        return torch.gather(all_hidden, dim=-2, index=positions)


class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(MLP, self).__init__()

        self._activator = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                        nn.Tanh(),
                                        nn.Linear(hidden_dim, output_dim))
        self._dropout = nn.Dropout(dropout_rate)

    def forward(self, var_h):
        return self._activator(self._dropout(var_h))


# class BiLSTM(nn.Module):
#     def __init__(self, args, label_vocab, lexical_vocab):
#         super(BiLSTM, self).__init__()
#
#         dropout = 0.4
#
#         self.UNK = '<UNK>'  # 未知字，padding符号
#         self.PAD = '<PAD>'
#
#         self._label_vocab = label_vocab
#         self._lexical_vocab = lexical_vocab
#         self.lexical_size = len(self._lexical_vocab)
#         self.label_size = len(self._label_vocab)
#         self.unk_id = self._lexical_vocab.index(self.UNK)
#
#         self.hidden_dim = args.hidden_dim
#         self._neg_rate = args.negative_rate
#
#         # 加载词向量
#         w_embedding = torch.tensor(np.load(args.embedding_file)['embeddings'])
#         self.word_embeds = nn.Embedding.from_pretrained(w_embedding, freeze=True)
#         self.embedding_dim = w_embedding.size()[-1]
#         self.dropout = nn.Dropout(dropout)
#         self.encoder = nn.LSTM(self.embedding_dim, self.hidden_dim // 2, num_layers=1, bidirectional=True,
#                                batch_first=True)
#         self._classifier = MLP(self.hidden_dim * 4, self.hidden_dim, self.label_size, dropout)
#         self._criterion = nn.NLLLoss()
#
#     def _pre_process_input(self, utterances):
#         lengths = [len(s) for s in utterances]
#         max_len = max(lengths)
#         units, masks, positions = [], [], []
#
#         for sentence in utterances:
#             units.append(sentence + [self.PAD] * (max_len - len(sentence)))
#             cum_list = np.cumsum([1] * len(sentence)).tolist()
#             positions.append(cum_list + [max_len - 1] * (max_len - len(sentence)))
#             masks.append([1] * len(sentence) + [0] * (max_len - len(sentence)))
#
#         var_unit = torch.LongTensor([[self._lexical_vocab.index(w, self.unk_id) for w in u] for u in units])
#         attn_mask = torch.LongTensor(masks)
#         var_start = torch.LongTensor(positions)
#
#         if torch.cuda.is_available():
#             var_unit = var_unit.cuda()
#             attn_mask = attn_mask.cuda()
#             var_start = var_start.cuda()
#         return var_unit, attn_mask, var_start, lengths
#
#     def _pre_process_output(self, entities, lengths):
#         positions, labels = [], []
#         batch_size = len(entities)
#
#         # positive samples set
#         for utt_i in range(0, batch_size):
#             for segment in entities[utt_i]:
#                 positions.append((utt_i, segment[0], segment[1]))
#                 labels.append(segment[2])
#
#         # sampling negative samples set
#         for utt_i in range(0, batch_size):
#             reject_set = [(e[0], e[1]) for e in entities[utt_i]]
#             s_len = lengths[utt_i]
#             neg_num = int(s_len * self._neg_rate) + 1
#
#             candies = flat_list([[(i, j) for j in range(i, s_len) if (i, j) not in reject_set] for i in range(s_len)])
#             if len(candies) > 0:
#                 sample_num = min(neg_num, len(candies))
#                 assert sample_num > 0
#
#                 np.random.shuffle(candies)
#                 for i, j in candies[:sample_num]:
#                     positions.append((utt_i, i, j))
#                     labels.append("O")
#
#         var_lbl = torch.LongTensor(iterative_support(self._label_vocab.index, labels))
#         if torch.cuda.is_available():
#             var_lbl = var_lbl.cuda()
#         return positions, var_lbl
#
#     def forward(self, var_sent, **kwargs):
#         embedding = self.word_embeds(var_sent)
#         embedding = self.dropout(embedding)
#         embedding = embedding.to(torch.float32)
#         outputs, hidden = self.encoder(embedding)
#
#         batch_size, token_num, hidden_dim = outputs.size()
#         ext_row = outputs.unsqueeze(2).expand(batch_size, token_num, token_num, hidden_dim)
#         ext_column = outputs.unsqueeze(1).expand_as(ext_row)
#         table = torch.cat([ext_row, ext_column, ext_row - ext_column, ext_row * ext_column], dim=-1)
#         score_t = self._classifier(table)
#         return score_t
#
#     def estimate(self, sentences, segments):
#         var_sent, attn_mask, start_mat, lengths = self._pre_process_input(sentences)
#         score_t = self(var_sent, mask_mat=attn_mask, starts=start_mat)
#         positions, targets = self._pre_process_output(segments, lengths)
#         flat_s = torch.cat([score_t[[i], j, k] for i, j, k in positions], dim=0)
#         return self._criterion(torch.log_softmax(flat_s, dim=-1), targets)
#
#     def inference(self, sentences):
#         var_sent, attn_mask, starts, lengths = self._pre_process_input(sentences)
#         log_items = self(var_sent, mask_mat=attn_mask, starts=starts)
#
#         score_t = torch.log_softmax(log_items, dim=-1)
#         val_table, idx_table = torch.max(score_t, dim=-1)
#
#         listing_it = idx_table.cpu().numpy().tolist()
#         listing_vt = val_table.cpu().numpy().tolist()
#         label_table = iterative_support(self._label_vocab.get, listing_it)
#
#         candidates = []
#         for l_mat, v_mat, sent_l in zip(label_table, listing_vt, lengths):
#             candidates.append([])
#             for i in range(0, sent_l):
#                 for j in range(i, sent_l):
#                     if l_mat[i][j] != "O":
#                         candidates[-1].append((i, j, l_mat[i][j], v_mat[i][j]))
#
#         entities = []
#         for segments in candidates:
#             ordered_seg = sorted(segments, key=lambda e: -e[-1])
#             filter_list = []
#             for elem in ordered_seg:
#                 flag = False
#                 current = (elem[0], elem[1])
#                 for prior in filter_list:
#                     flag = conflict_judge(current, (prior[0], prior[1]))
#                     if flag:
#                         break
#                 if not flag:
#                     filter_list.append((elem[0], elem[1], elem[2]))
#             entities.append(sorted(filter_list, key=lambda e: e[0]))
#         return entities
#
#
# if __name__ == '__main__':
#     b = torch.Tensor([[1, 2, 3], [4, 5, 6]])
#     print(b)
#     index_1 = torch.LongTensor([[0, 1], [2, 0]])
#     # index_2 = torch.LongTensor([[0, 1, 1], [0, 0, 0]])
#     y1 = torch.gather(b, dim=1, index=index_1)
#     # y2 = torch.gather(b, dim=0, index=index_2)
#     print(y1)
#     # print(y2)

