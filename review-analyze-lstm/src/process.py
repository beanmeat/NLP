import pandas as pd
from sklearn.model_selection import train_test_split

import config
from tokenizer import JiebaTokenizer


def process():
    print('开始数据预处理')
    # 读取
    df = pd.read_csv(config.RAW_DATA_DIR / 'online_shopping_10_cats.csv', usecols=['review', 'label'], encoding='utf-8')

    # 过滤数据
    df = df.dropna()

    # 划分数据集
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'])

    # 构建词表
    JiebaTokenizer.build_vocab(train_df['review'], config.PROCESSED_DIR / 'vocab.txt')

    # 构建tokenizer
    tokenizer = JiebaTokenizer.from_vocab(config.PROCESSED_DIR / 'vocab.txt')

    # 构建训练集
    train_df['review'] = train_df['review'].apply(lambda x: tokenizer.encode(x,config.SEQ_LEN))

    # 计算序列长度（95%分位数）
    # print(train_df['review'].apply(lambda x: len(x)).quantile(0.95))

    # 保存训练集
    train_df.to_json(config.PROCESSED_DIR / 'indexed_train.jsonl', orient='records', lines=True)

    # 构建测试机
    test_df['review'] = test_df['review'].apply(lambda x: tokenizer.encode(x,config.SEQ_LEN))
    # 保存测试机
    test_df.to_json(config.PROCESSED_DIR / 'indexed_test.jsonl', orient='records', lines=True)

    print('数据预处理完成')


if __name__ == '__main__':
    process()
