"""
@Project ：SIFRank_zh
@File ：config.py
@IDE ：PyCharm
"""
from ltp import LTP
from transformers import BertConfig

from SIFRank.config import cfg
from SIFRank.embeddings.sent_emb_sif import SentEmbeddings
from SIFRank.embeddings.word_emb_roberta import BERTClassifier
from SIFRank.model.method import SIFRank


bert_config = BertConfig.from_pretrained(cfg.model_name_or_path)
emb_model = BERTClassifier(bert_config)
SIF = SentEmbeddings(cfg, emb_model)
ltp = LTP()


def extract(text):
    """

    :param text:
    :return:
    """
    keywords_roberta = SIFRank(text,
                               SIF,
                               ltp,
                               N=cfg.num_keywords,
                               sent_emb_method=cfg.sent_emb_method,
                               elmo_layers_weight=cfg.elmo_layers_weight
                               )
    print("------------------------------------------")
    print("原文:" + text)
    # print("------------------------------------------")
    print("SIFRank roberta 结果:")
    print(keywords_roberta)
    return keywords_roberta


if __name__ == '__main__':

    text = '红米2a维修要怎么做'
    extract(text)
