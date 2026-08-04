import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import config


# 1. 定义Dataset
class ReviewAnalyzeDataset(Dataset):
    def __init__(self, data_path):
        # [{"label":1,"review":[1032,3153,...0,0,0,0,0]},{"label":1,"review":[1032,3153,...0,0,0,0,0]}]
        self.data = pd.read_json(data_path, orient='records', lines=True).to_dict(orient='records')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_tensor = torch.tensor(self.data[index]['review'], dtype=torch.long)
        target_tensor = torch.tensor(self.data[index]['label'], dtype=torch.float32)
        return input_tensor, target_tensor


# 2. 获取DataLoader得方法
def get_dataloader(train=True):
    data_path = config.PROCESSED_DIR / 'indexed_train.jsonl' if train else config.PROCESSED_DIR / 'indexed_test.jsonl'
    dataset = ReviewAnalyzeDataset(data_path)
    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

if __name__ == '__main__':
    train_dataloader = get_dataloader(train=True)
    print(f'train batch个数：{len(train_dataloader)}')
    test_dataloader = get_dataloader(train=False)
    print(f'test batch个数：{len(test_dataloader)}')

    for inputs,targets in train_dataloader:
        print(f'inputs.shape:{inputs.shape}') # [batch_size, seq_len]
        print(f'targets.shape:{targets.shape}') # [batch_size]
        break