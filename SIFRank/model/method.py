import numpy as np
from collections import defaultdict

from SIFRank.model.input_representation import InputTextObj
from SIFRank.config import stop_words


def cos_sim(vector_a, vector_b):
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a
    :param vector_b: 向量 b
    :return: sim
    """
    vector_a = vector_a.cpu()
    vector_b = vector_b.cpu()
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    if denom == 0.0:
        return 0.0
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim


def get_dist_cosine(emb1, emb2, sent_emb_method, elmo_layers_weight):
    sum = 0.0
    assert emb1.shape == emb2.shape
    if sent_emb_method == "elmo":
        for i in range(0, 3):
            sum += cos_sim(emb1[i], emb2[i]) * elmo_layers_weight[i]
    elif sent_emb_method == 'roberta':
        sum += cos_sim(emb1, emb2)
    return sum


def get_all_dist(candidate_embeddings_list, text_obj, dist_list):
    '''
    :param candidate_embeddings_list:
    :param text_obj:
    :param dist_list:
    :return: dist_all
    '''
    dist_all = defaultdict(list)
    for i, emb in enumerate(candidate_embeddings_list):
        phrase = text_obj.keyphrase_candidate[i][0]
        dist_all[phrase].append(dist_list[i])
    return dist_all


def get_final_dist(dist_all, method="average"):
    '''
    :param dist_all:
    :param method: "average"
    :return:
    '''

    final_dist = {}

    if method == "average":
        for phrase, dist_list in dist_all.items():
            sum_dist = 0.0
            for dist in dist_list:
                sum_dist += dist
            if phrase in stop_words:
                sum_dist = 0.0
            final_dist[phrase] = sum_dist / float(len(dist_list))

    return final_dist


def softmax(x):
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x


def get_position_score(keyphrase_candidate_list, position_bias):
    # length = len(keyphrase_candidate_list)
    position_score = {}
    for i, kc in enumerate(keyphrase_candidate_list):
        np = kc[0]
        if np in position_score:
            position_score[np] += 0.0
        else:
            position_score[np] = 1 / (float(i) + 1 + position_bias)
    score_list = []
    for np, score in position_score.items():
        score_list.append(score)
    score_list = softmax(score_list)

    i = 0
    for np, score in position_score.items():
        position_score[np] = score_list[i]
        i += 1
    return position_score


def SIFRank(text, SIF, zh_model, N, sent_emb_method, elmo_layers_weight):
    """
    :param text:
    :param SIF: sent_embeddings
    :param zh_model:
    :param N: the top-N number of keyphrases
    :param sent_emb_method: 'elmo', 'glove'
    :param elmo_layers_weight: the weights of different layers of ELMo
    :return:
    """
    text_obj = InputTextObj(zh_model, text)
    sent_embeddings, candidate_embeddings_list = SIF.get_tokenized_sent_embeddings(text_obj)
    dist_list = []
    for i, emb in enumerate(candidate_embeddings_list):
        dist = get_dist_cosine(sent_embeddings, emb, sent_emb_method, elmo_layers_weight=elmo_layers_weight)
        dist_list.append(dist)
    dist_all = get_all_dist(candidate_embeddings_list, text_obj, dist_list)
    dist_final = get_final_dist(dist_all, method='average')
    dist_sorted = sorted(dist_final.items(), key=lambda x: x[1], reverse=True)
    return dist_sorted[0:N]
