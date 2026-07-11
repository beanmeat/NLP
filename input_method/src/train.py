import time

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import config
from input_method.src.dataset import get_dataloader
from input_method.src.model import InputMethodModel


def train_one_epoch(model, dataloader, loss_function, optimizer, device):
    """
    训练一个轮次
    :param model: 模型
    :param dataloader: 数据加载器
    :param loss_function: 损失函数
    :param optimizer: 优化器
    :param device: 设备
    :return: 每个batch的平均loss
    """
    model.train()
    epoch_total_loss = 0
    for inputs, targets in tqdm(dataloader,desc='训练'):
        # inputs.shape: [batch_size, seq_len]
        # targets.shape: [batch_size]
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        # outputs.shape: [batch_size, vocab_size]
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()
        epoch_total_loss += loss.item()
    return epoch_total_loss / len(dataloader)


def train():
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'设备：{device}')

    # 数据
    dataloader = get_dataloader()
    print("数据集加载完成")

    # 加载词表
    with open(config.PROCESSED_DIR / 'vocab.txt', 'r', encoding='utf-8') as f:
        vocab_list = [line[:-1] for line in f.readlines()]
    print("词表加载完成")

    # 模型
    model = InputMethodModel(vocab_size=len(vocab_list)).to(device)

    # 损失函数
    loss_function = torch.nn.CrossEntropyLoss()

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # tensorboard write
    writer = SummaryWriter(config.LOGS_DIR/time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime()))

    # 开始训练
    best_loss = float('inf')
    for epoch in range(1, config.EPOCHS + 1):
        print(f'======= Epoch {epoch} =======')
        # 训练一轮的逻辑
        avg_loss = train_one_epoch(model, dataloader, loss_function, optimizer,device)
        print(f'Loss: {avg_loss:.4f}')

        # 记录训练结果
        writer.add_scalar('Loss', avg_loss, epoch)

        # 保存模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), config.MODELS_DIR / 'model.pt')
            print('模型保存成功')
    return True


if __name__ == '__main__':
    train()
