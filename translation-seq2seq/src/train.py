import torch
from tokenizer import ChineseTokenizer,EnglishTokenizer
import config
from model import TranslationDecoder,TranslationEncoder
from dataset import get_dataloader


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # tokenizer
    zh_tokenizer = ChineseTokenizer.from_vocab(config.PROCESSED_DIR/'zh_vocab.txt')
    en_tokenizer = EnglishTokenizer.from_vocab(config.PROCESSED_DIR/'en_vocab.txt')

    # 模型
    encoder = TranslationEncoder(zh_tokenizer.vocab_size,zh_tokenizer.pad_token_id)
    decoder = TranslationDecoder(en_tokenizer.vocab_size,en_tokenizer.pad_token_id)

    # 加载数据
    dataloader = get_dataloader()

    # 损失函数
    loss_function = torch.nn.CrossEntropyLoss(ignore_index=en_tokenizer.pad_token_id)