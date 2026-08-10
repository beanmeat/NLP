import pandas as pd
import torch
from datasets import load_from_disk
from torch.utils.data import Dataset, DataLoader
import config



# 2. 获取DataLoader得方法
def get_dataloader(train=True):
    data_path = str(config.PROCESSED_DATA_DIR / ('train' if train else 'test'))
    dataset = load_from_disk(data_path)
    dataset.set_format(type='torch')
    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

if __name__ == '__main__':
    train_dataloader = get_dataloader(train=True)
    print(f'train batch个数：{len(train_dataloader)}')
    test_dataloader = get_dataloader(train=False)
    print(f'test batch个数：{len(test_dataloader)}')

    for batch in train_dataloader:
        print(f'input_ids.shape:{batch['input_ids'].shape}')
        print(f'attention_mask.shape:{batch['attention_mask'].shape}')
        print(f'label.shape:{batch['label'].shape}')
        break