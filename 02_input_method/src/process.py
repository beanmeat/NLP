import jieba
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import config
from input_method.src.tokenizer import JiebaTokenizer


def build_dataset(sentences, tokenizer):
    """
    构建并保存数据集
    :param sentences: 原始句子列表
    :param tokenizer: 分词器对象
    :return: [{input:[1,2,3,4,5],target:6},{input:[1,2,3,4,5],target:6}]
    """
    index_sentences = [tokenizer.encode(sentence) for sentence in sentences]
    dataset = []
    for sentence in index_sentences:
        # sentence: [1,2,3,4,5,6,7,8,9,10]
        for i in range(len(sentence) - config.SEQ_LEN):
            input = sentence[i:i + config.SEQ_LEN]
            target = sentence[i + config.SEQ_LEN]
            dataset.append({'input': input, 'target': target})
    return dataset


def process():
    """
    预处理数据
    """
    print('开始处理数据...')
    # 读取数据
    df = pd.read_json(config.ROW_DATA_DIR / 'synthesized_.jsonl', orient='records', lines=True).sample(frac=0.03)

    # 抽取句子
    sentences = []
    for dialog in df['dialog']:
        for sentence in dialog:
            sentences.append(sentence.split('：')[1])
    print(f'句子总数：{len(sentences)}')

    # 划分数据集
    train_sentences, test_sentences = train_test_split(sentences, test_size=0.2)
    print(f'训练集句子数：{len(train_sentences)}')
    print(f'测试集句子数：{len(test_sentences)}')

    # 构建词表（用训练集）
    vocab_list = JiebaTokenizer.build_vocab(train_sentences, config.PROCESSED_DIR / 'vocab.txt')

    # 创建tokenizer
    tokenizer = JiebaTokenizer.from_vocab(config.PROCESSED_DIR / 'vocab.txt')

    # 构建训练集
    train_dataset = build_dataset(train_sentences, tokenizer)

    # 保存训练集
    pd.DataFrame(train_dataset).to_json(config.PROCESSED_DIR / 'index_train.jsonl', lines=True, orient='records')

    # 构建并保存测试集
    test_dataset = build_dataset(test_sentences, tokenizer)

    # 保存测试集
    pd.DataFrame(test_dataset).to_json(config.PROCESSED_DIR / 'index_test.jsonl', lines=True, orient='records')

    print('数据处理完成...')


if __name__ == '__main__':
    print(__file__)
    process()
