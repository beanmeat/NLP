import torch
from datasets import tqdm

import config
from dataset import get_dataloader
from predict import predict_batch
from model import ReviewAnalyzeModel


def evaluate_model(model, dataloader, device):
    total_count = 0
    correct_count = 0
    model.eval()
    for batch in tqdm(dataloader, desc='评估'):
        input_ids = batch['input_ids'].to(device)  # inputs.shape: [batch_size, seq_len]
        attention_mask = batch['attention_mask'].to(device)
        targets = batch['label'].tolist()

        outputs = predict_batch(input_ids, attention_mask, model)  # outputs.shape: [batch_size]

        for output, target in zip(outputs, targets):
            output = 1 if output > 0.5 else 0
            total_count += 1
            if output == target:
                correct_count += 1
    return correct_count / total_count


def run_evaluate():
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 模型
    model = ReviewAnalyzeModel().to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR / 'model.pt'))
    # 数据集
    dataloader = get_dataloader(train=False)
    # 模型评估
    acc = evaluate_model(model, dataloader, device)
    print(f'准确率：{acc:.4f}')


if __name__ == '__main__':
    run_evaluate()
