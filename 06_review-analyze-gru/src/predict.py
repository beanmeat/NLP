import torch

import config
from tokenizer import JiebaTokenizer
from model import ReviewAnalyzeModel


def predict_batch(model, input_tensor):
    """
    批量预测
    :param model: 模型
    :param input_tensor: 输入张量[batch_size,seq_len]
    :return: 一批预测结果 [0.5,0.7,0.9]
    """
    model.eval()
    with torch.no_grad():
        output = model(input_tensor) # output.shape: [batch_size]
        return torch.sigmoid(output).tolist()


def predict(user_input, model, tokenizer, device):
    # 处理输入
    index_list = tokenizer.encode(user_input,config.SEQ_LEN)
    input_tensor = torch.tensor([index_list], dtype=torch.long).to(device)
    # input_tensor.shape: [batch_size, seq_len]
    batch_result = predict_batch(model,input_tensor)
    return batch_result[0]


def run_predict():
    # 准备资源
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # tokenizer
    tokenizer = JiebaTokenizer.from_vocab(config.PROCESSED_DIR / 'vocab.txt')
    # 模型
    model = ReviewAnalyzeModel(vocab_size=tokenizer.vocab_size, padding_index=tokenizer.pad_token_id).to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR / 'model.pt'))

    while True:
        user_input = input('>')
        if user_input in ['q', 'quit']:
            print('程序已退出')
            break
        if user_input.strip() == '':
            continue
        result = predict(user_input, model, tokenizer, device)
        if result > 0.5:
            print(f'正向评价（置信度：{result:.2f}）')
        else:
            print(f'负向评价（置信度：{1 - result:.2f}）')


if __name__ == '__main__':
    run_predict()
