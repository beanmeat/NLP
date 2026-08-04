import time

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import config
from dataset import get_dataloader
from tokenizer import JiebaTokenizer
from model import ReviewAnalyzeModel


def train_one_epoch(model, dataloader, loss_function, optimizer, device):
    model.train()
    epoch_total_loss = 0
    for inputs,targets in tqdm(dataloader,desc='训练'):
        # inputs.shape: [batch_size, seq_len]
        # targets.shape: [batch_size]
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        # 前向传播
        outputs = model(inputs) # outputs.shape: [batch_size]
        # 计算损失
        loss = loss_function(outputs, targets)
        # 反向传播
        loss.backward()
        # 优化
        optimizer.step()
        epoch_total_loss += loss.item()
    return epoch_total_loss / len(dataloader)



def train():
    # 选择设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Tokenizer
    tokenizer = JiebaTokenizer.from_vocab(config.PROCESSED_DIR / 'vocab.txt')
    # 模型
    model = ReviewAnalyzeModel(vocab_size=tokenizer.vocab_size, padding_index=tokenizer.pad_token_id).to(device)
    # 数据
    dataloader = get_dataloader()
    # 损失函数
    loss_function = torch.nn.BCEWithLogitsLoss()
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    # tensorboard
    writer = SummaryWriter(config.LOGS_DIR / time.strftime('%Y-%m-%d_%H-%M-%S'))

    best_loss = float('inf')
    for epoch in range(1, config.EPOCHS + 1):
        print(f'======= Epoch {epoch} =======')
        avg_loss = train_one_epoch(model, dataloader, loss_function, optimizer, device)
        print(f'Loss: {avg_loss:.4f}')
        writer.add_scalar('Loss', avg_loss, epoch)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), config.MODELS_DIR / 'model.pt')
            print('模型保存成功')


if __name__ == '__main__':
    train()
