from SIFRank.config import stop_words, considered_tags
from SIFRank.model import extractor


class InputTextObj:
    """Represent the input text in which we want to extract keyphrases"""

    def __init__(self, zh_model, text=""):
        """
        :param zh_model: the pipeline of Chinese tokenization and POS-tagger
        :param text:
        """
        text = text.replace('#', '')
        seg, hidden = zh_model.seg([text])
        pos = zh_model.pos(hidden)
        self.tokens = [w for w in seg[0]]
        self.tokens_tagged = [(word, pos) for word, pos in zip(seg[0], pos[0])]
        assert len(self.tokens) == len(self.tokens_tagged)

        for i, token in enumerate(self.tokens):
            if token in stop_words:
                self.tokens_tagged[i] = (token, "u")
            if token == '-':
                self.tokens_tagged[i] = (token, "-")

        self.keyphrase_candidate = extractor.extract_candidates(self.tokens_tagged, zh_model)
