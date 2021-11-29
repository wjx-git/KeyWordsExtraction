"""
@Project ：SIFRank
@File ：word_emb_roberta.py
@IDE ：PyCharm
"""
import torch
from transformers import BertModel, BertPreTrainedModel, BertTokenizer

from SIFRank.config import cfg as args


class BERTClassifier(BertPreTrainedModel):
    def __init__(self, bert_config):
        super(BERTClassifier, self).__init__(bert_config)
        self.tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
        self.bert = BertModel(config=bert_config)
        self.bert.to('cuda' if torch.cuda.is_available() else 'cpu')
        self.bert.eval()

    def get_tokenized_words_embeddings(self, tokens):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        text = ''.join(tokens[0])
        encoded_dict = self.tokenizer(text, return_tensors="pt")
        input_ids = encoded_dict['input_ids'].to(device)
        token_type_ids = encoded_dict['token_type_ids'].to(device)
        mask = encoded_dict['attention_mask'].to(device)
        outputs = self.bert(input_ids, attention_mask=mask, token_type_ids=token_type_ids)
        last_hidden_state = outputs[0].squeeze()
        shape = last_hidden_state.size()
        words_ids = self.words_position(text, tokens[0])
        embeddings = torch.zeros((len(tokens[0]), shape[-1])).to(device)
        for k, w_id in enumerate(words_ids):
            e = torch.zeros((1, shape[-1])).to(device)
            for i in w_id:
                e += last_hidden_state[i+1]
            embeddings[k] = e / len(w_id)
        embeddings = embeddings.unsqueeze(0).unsqueeze(0)
        return embeddings.detach()

    def words_position(self, text, words):
        tokens = self.tokenizer.tokenize(text)
        positions = []  # 合并被拆开的token
        tp = []
        for i, token in enumerate(tokens):
            if not token.startswith('##'):
                if tp:
                    positions.append(tp)
                tp = [i]
            else:
                tp.append(i)
        if tp:
            positions.append(tp)

        wp = []
        start = 0
        if positions:
            for w in words:
                twp = []
                length = 0
                while length < len(w):
                    twp.extend(positions[start])
                    for p in positions[start]:
                        length += len(tokens[p].lstrip('##'))
                    start += 1
                wp.append(twp)
        return wp

