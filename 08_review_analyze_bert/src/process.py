from datasets import load_dataset, ClassLabel
from transformers import AutoTokenizer

import config


def process():
    print('开始数据预处理')
    # 读取
    print(config.RAW_DATA_DIR / 'online_shopping_10_cats.csv')
    dataset = load_dataset('csv', data_files=str(config.RAW_DATA_DIR / 'online_shopping_10_cats.csv'))['train']
    # 过滤数据
    dataset = dataset.remove_columns(['cat'])
    dataset = dataset.filter(lambda x: x['review'] is not None)
    # print(dataset.features)

    # 划分数据集
    dataset = dataset.cast_column('label', ClassLabel(names=['消极', '积极']))
    # print(dataset.features)
    dataset_dict = dataset.train_test_split(test_size=0.2, stratify_by_column='label')  # 分层抽样
    # print(dataset_dict)

    # 构建tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.PRETRAINED_MODELS_DIR / 'bert-base-chinese')

    # 构建训练集
    def tokenize(batch):
        tokenized = tokenizer(batch['review'], truncation=True, padding="max_length", max_length=config.SEQ_LEN)
        return {'input_ids': tokenized['input_ids'],'attention_mask': tokenized['attention_mask'],'label': batch['label']}

    dataset_dict = dataset_dict.map(tokenize, batched=True, remove_columns=['review'])
    print(dataset_dict)

    # 保存训练集

    # 构建测试机

    # 保存测试机

    print('数据预处理完成')


if __name__ == '__main__':
    process()
