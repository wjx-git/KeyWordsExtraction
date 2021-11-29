import numpy
import torch

from SIFRank.config import english_punctuations, chinese_punctuations, stop_words, considered_tags


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class SentEmbeddings:

    def __init__(self, args, word_embedder):
        self.word2weight_pretrain = get_word_weight(args.weightfile_pretrain, args.weightpara_pretrain)
        self.word2weight_finetune = get_word_weight(args.weightfile_finetune, args.weightpara_finetune)
        self.word_embeddor = word_embedder
        self.lambda_ = args.lambda_
        self.database = args.database
        self.embeddings_type = args.embeddings_type
        self.if_ds = args.if_DS
        self.if_ea = args.if_EA

    def get_tokenized_sent_embeddings(self, text_obj):
        """
        Based on part of speech return a list of candidate phrases
        :param text_obj: Input text Representation see @InputTextObj
        """
        embeddings = self.word_embeddor.get_tokenized_words_embeddings([text_obj.tokens])

        candidate_embeddings_list = []
        weight_list = get_weight_list(self.word2weight_pretrain, self.word2weight_finetune, text_obj.tokens,
                                      lamda=self.lambda_, database=self.database)

        tokens_id = [i for i in range(len(text_obj.tokens))]
        sent_embeddings = get_weighted_average(text_obj.tokens, tokens_id, weight_list, embeddings[0],
                                               embeddings_type=self.embeddings_type,
                                               sents_tokened_tagged=text_obj.tokens_tagged)

        for kc in text_obj.keyphrase_candidate:
            candidate_tokens_id = [i for i in range(kc[1][0], kc[1][1])]
            kc_emb = get_weighted_average(text_obj.tokens, candidate_tokens_id, weight_list, embeddings[0],
                                          embeddings_type=self.embeddings_type)
            candidate_embeddings_list.append(kc_emb)

        return sent_embeddings, candidate_embeddings_list


def mat_division(vector_a, vector_b):
    a = vector_a.detach().numpy()
    b = vector_b.detach().numpy()
    A = numpy.mat(a)
    B = numpy.mat(b)
    return torch.from_numpy(numpy.dot(A.I, B))


def get_sent_segmented(tokens):
    min_seq_len = 16
    sents_sectioned = []
    if len(tokens) <= min_seq_len:
        sents_sectioned.append(tokens)
    else:
        position = 0
        for i, token in enumerate(tokens):
            if token == '.' or token == '。':
                if i - position >= min_seq_len:
                    sents_sectioned.append(tokens[position:i + 1])
                    position = i + 1
        if len(tokens[position:]) > 0:
            sents_sectioned.append(tokens[position:])

    return sents_sectioned


def splice_embeddings(elmo_embeddings, tokens_segmented):
    new_elmo_embeddings = elmo_embeddings[0:1, :, 0:len(tokens_segmented[0]), :]
    for i in range(1, len(tokens_segmented)):
        emb = elmo_embeddings[i:i + 1, :, 0:len(tokens_segmented[i]), :]
        new_elmo_embeddings = torch.cat((new_elmo_embeddings, emb), 2)
    return new_elmo_embeddings


def get_effective_words_num(tokened_sents):
    num = sum([1 if token not in english_punctuations else 0 for token in tokened_sents])
    return num


def get_weighted_average(tokenized_sents, tokened_id, weight_list, embeddings_list, embeddings_type="elmo",
                         sents_tokened_tagged=None):
    assert len(tokenized_sents) == len(weight_list)
    num_words = len(tokened_id)
    shape = embeddings_list.size()
    if embeddings_type == "elmo" or embeddings_type == "elmo_sectioned":
        sum = torch.zeros((3, shape[-1]))  # 每一层词向量叠加得到每层的句向量
        layers = 3
    elif embeddings_type == "roberta":
        sum = torch.zeros((1, shape[-1]))
        layers = 1
    elif embeddings_type == "glove":
        sum = numpy.zeros((1, embeddings_list.shape[2]))
        layers = 1
    else:
        raise Exception('embedding_type error')
    sum.to(device)
    for i in range(0, layers):
        for j in tokened_id:
            if not sents_tokened_tagged:
                e_test = embeddings_list[i][j]
                sum[i] += e_test * weight_list[j]
            elif sents_tokened_tagged[j][1] in considered_tags:
                e_test = embeddings_list[i][j]
                sum[i] += e_test * weight_list[j]
        sum[i] = sum[i] / float(num_words)
    return sum


def get_weight(tokenized_sents, word2weight, word, method="max_weight"):
    if word in word2weight:
        return word2weight[word]
    if (word in stop_words) or (word in english_punctuations) or (word in chinese_punctuations):
        return 0.0
    if method == "max_weight":  # Return the max weight of word in the tokenized_sents, for oov words
        return max([word2weight.get(w, 0.0) for w in tokenized_sents])
    return 0.0


def get_weight_list(word2weight_pretrain, word2weight_finetune, tokenized_sents, lamda, database):
    weight_list = []

    for word in tokenized_sents:
        if database is None:
            weight = get_weight(tokenized_sents, word2weight_pretrain, word, method="max_weight")
        else:
            weight_pretrain = get_weight(tokenized_sents, word2weight_pretrain, word, method="max_weight")
            weight_finetune = get_weight(tokenized_sents, word2weight_finetune, word, method="max_weight")
            weight = lamda * weight_pretrain + (1.0 - lamda) * weight_finetune
        weight_list.append(weight)
    if device != 'cpu':
        weight_list = torch.tensor(weight_list, dtype=torch.float).to(device)

    return weight_list


def get_normalized_weight(weight_list):
    sum_weight = sum([w for w in weight_list])
    if sum_weight == 0.0:
        return weight_list
    for i in range(len(weight_list)):
        weight_list[i] /= sum_weight
    return weight_list


def get_word_weight(weightfile, weightpara=2.7e-4):
    """
    Get the weight of words by word_fre/sum_fre_words
    :param weightfile
    :param weightpara
    :return: word2weight[word]=weight : a dict of word weight
    """
    if weightpara <= 0:  # when the parameter makes no sense, use unweighted
        weightpara = 1.0
    word2weight = {}
    word2fre = {}
    with open(weightfile, encoding='UTF-8') as f:
        lines = f.readlines()
    sum_fre_words = 0
    for line in lines:
        word_fre = line.split()
        if len(word_fre) >= 2:
            word2fre[word_fre[0]] = float(word_fre[1])
            sum_fre_words += float(word_fre[1])
        else:
            print(line)
    for key, value in word2fre.items():
        word2weight[key] = weightpara / (weightpara + value / sum_fre_words)
    return word2weight
