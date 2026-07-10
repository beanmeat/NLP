import jieba
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import config


def build_dataset(sentences, word2index):
    """
    构建并保存数据集
    :param sentences: 原始句子列表
    :param word2index: {word:index}
    :return: [{input:[1,2,3,4,5],target:6},{input:[1,2,3,4,5],target:6}]
    """
    index_sentences = [[word2index.get(word, 0) for word in jieba.lcut(sentence)] for sentence in sentences]
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
    df = pd.read_json(config.ROW_DATA_DIR / 'synthesized_.jsonl', orient='records', lines=True).sample(frac=0.1)

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
    vocab_set = set()
    for sentence in tqdm(train_sentences, desc='构建词表'):
        for word in jieba.lcut(sentence):
            vocab_set.add(word)
    vocab_list = ['<unk>'] + list(vocab_set)
    print(f'词表大小：{len(vocab_list)}')

    word2index = {word: index for index, word in enumerate(vocab_list)}

    # 构建训练集
    train_dataset = build_dataset(train_sentences, word2index)
    # 保存训练集
    pd.DataFrame(train_dataset).to_json(config.PROCESSED_DIR / 'index_train.jsonl', lines=True, orient='records')
    # 构建并保存测试机
    test_dataset = build_dataset(test_sentences, word2index)
    pd.DataFrame(test_dataset).to_json(config.PROCESSED_DIR / 'index_test.jsonl', lines=True, orient='records')
    print('数据处理完成...')


if __name__ == '__main__':
    print(__file__)
    process()
