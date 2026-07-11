import torch

from input_method.src import config
from input_method.src.dataset import get_dataloader
from input_method.src.model import InputMethodModel
from predict import predict_batch

def evaluate_model(model, dataloader, device):
    total_count = 0
    top1_acc_count = 0
    top5_acc_count = 0
    for inputs,targets in dataloader:
        inputs.to(device)
        targets = targets.tolist() # targets = [5,8,13]
        top5_indexes_list = predict_batch(model,inputs) # top5_indexes_list = [[1,2,3,4,5],[2,3,4,5,6],[3,4,5,6,7]]

        for target,top5_indexes in zip(targets,top5_indexes_list):
            total_count += 1
            if target in top5_indexes:
                top5_acc_count += 1
            if top5_indexes[0] == target:
                top1_acc_count += 1

    return top1_acc_count / total_count, top5_acc_count / total_count

def run_evaluate():
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 词表
    with open(config.PROCESSED_DIR / 'vocab.txt', 'r', encoding='utf-8') as f:
        vocab_list = [line[:-1] for line in f.readlines()]
    print("词表加载完成")
    # 模型
    model = InputMethodModel(vocab_size=len(vocab_list)).to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR / 'model.pt'))

    # 数据集
    dataloader = get_dataloader(train=False)
    # 评估模型
    top1_acc,top5_acc = evaluate_model(model,dataloader, device)
    print(f'top1_acc:{top1_acc:.4f},top5_acc:{top5_acc:.4f}')

if __name__ == '__main__':
    run_evaluate()