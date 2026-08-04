import jieba
from tqdm import tqdm
import config


class JiebaTokenizer:
    unk_token = '<unk>'
    pad_token = '<pad>'

    def __init__(self, vocab_list):
        self.vocab_list = vocab_list
        self.vocab_size = len(vocab_list)

        self.word2index = {word: index for index, word in enumerate(vocab_list)}
        self.index2word = {index: word for index, word in enumerate(vocab_list)}

        self.unk_token_id = self.word2index.get(self.unk_token)
        self.pad_token_id = self.word2index.get(self.pad_token)

    @staticmethod
    def tokenize(text):
        return jieba.lcut(text)

    def encode(self, text, seq_len):
        word_list = self.tokenize(text)
        # 补齐或截断到指定的seq_len
        if len(word_list) > seq_len:
            word_list = word_list[0:seq_len]
        elif len(word_list) < seq_len:
            word_list = word_list + [self.pad_token] * (seq_len - len(word_list))
        return [self.word2index.get(word, self.unk_token_id) for word in word_list]

    @classmethod
    def from_vocab(cls, vocab_file):
        # 1. 加载词表文件
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab_list = [line[:-1] for line in f.readlines()]

        # 2. 创建tokenizer对象
        return cls(vocab_list)

    @classmethod
    def build_vocab(cls, sentences, vocab_file):
        # 构建词表（用训练集）
        vocab_set = set()
        for sentence in tqdm(sentences, desc='构建词表'):
            for word in jieba.lcut(sentence):
                if word.strip() != '':  # 去除不可见的token
                    vocab_set.add(word)
        vocab_list = [cls.pad_token, cls.unk_token] + list(vocab_set)
        print(f'词表大小：{len(vocab_list)}')

        # 保存词表
        with open(vocab_file, 'w', encoding='utf-8') as f:
            for word in vocab_list:
                f.write(word + '\n')
        print('词表保存完成')


if __name__ == '__main__':
    tokenizer = JiebaTokenizer.from_vocab(config.PROCESSED_DIR / 'vocab.txt')
    print(tokenizer.encode('我喜欢坐地铁',config.SEQ_LEN))
