"""
@Project ：SIFRank_zh
@File ：config.py
@IDE ：PyCharm
"""


def _load_stopwords(stopwords_file):
    words = set()
    with open(stopwords_file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            words.add(line.rstrip())
    return words


class Config:
    considered_tags = {'n', 'np', 'ns', 'ni', 'nz', 'a', 'd', 'i', 'j', 'x', 'g'}
    english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
    chinese_punctuations = '！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.'
    stopwords = _load_stopwords('SIFRank/auxiliary_data/chinese_stopwords.txt')
    # model_file = r'auxiliary_data/zhs.model/'
    elmo_layers_weight = [0.0, 1.0, 0.0]
    num_keywords = 1  # the top-N number of keyphrases
    weightfile_pretrain = 'SIFRank/auxiliary_data/dict.txt'
    weightfile_finetune = 'SIFRank/auxiliary_data/dict.txt'
    weightpara_pretrain = 2.7e-4
    weightpara_finetune = 2.7e-4
    lambda_ = 1.0
    database = None
    embeddings_type = "roberta"  # "roberta"
    sent_emb_method = "roberta"  # "roberta"
    if_DS = False  # if take document segmentation(DS)
    if_EA = False  # if take  embeddings alignment(EA)
    model_name_or_path = 'SIFRank/auxiliary_data/pre_trained_model/bert'


cfg = Config()
english_punctuations = cfg.english_punctuations
chinese_punctuations = cfg.chinese_punctuations
stop_words = cfg.stopwords
considered_tags = cfg.considered_tags

