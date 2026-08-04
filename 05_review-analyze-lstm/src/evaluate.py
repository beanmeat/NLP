import torch

import config
from dataset import get_dataloader
from predict import predict_batch
from tokenizer import JiebaTokenizer
from model import ReviewAnalyzeModel


def evaluate_model(model, dataloader, device):
    total_count = 0
    correct_count = 0
    model.eval()
    for inputs,targets in dataloader:
        inputs = inputs.to(device) # inputs.shape: [batch_size, seq_len]
        targets = targets.tolist()
        outputs = predict_batch(model,inputs) # outputs.shape: [batch_size]

        for output,target in zip(outputs,targets):
            output = 1 if output > 0.5 else 0
            total_count += 1
            if output == target:
                correct_count += 1
    return correct_count / total_count



def run_evaluate():
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 创建tokenizer
    tokenizer = JiebaTokenizer.from_vocab(config.PROCESSED_DIR / 'vocab.txt')
    # 模型
    model = ReviewAnalyzeModel(vocab_size=tokenizer.vocab_size,padding_index=tokenizer.pad_token_id).to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR / 'model.pt'))
    # 数据集
    dataloader = get_dataloader(train=False)
    # 模型评估
    acc = evaluate_model(model, dataloader, device)
    print(f'准确率：{acc:.4f}')


if __name__ == '__main__':
    run_evaluate()