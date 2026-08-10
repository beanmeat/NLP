import torch
from transformers import AutoTokenizer

import config
from model import ReviewAnalyzeModel


def predict_batch(input_ids,attention_mask,model):

    model.eval()
    with torch.no_grad():
        output = model(input_ids,attention_mask)
        return torch.sigmoid(output).tolist()


def predict(user_input, model, tokenizer, device):
    # 处理输入
    tokenized = tokenizer([user_input],padding='max_length',truncation=True,max_length=config.SEQ_LEN,return_tensors='pt')
    input_ids = tokenized['input_ids'].to(device)
    attention_mask = tokenized['attention_mask'].to(device)
    batch_result = predict_batch(input_ids,attention_mask,model)
    return batch_result[0]


def run_predict():
    # 准备资源
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(config.PRETRAINED_MODELS_DIR / 'bert-base-chinese'))
    # 模型
    model = ReviewAnalyzeModel().to(device)
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
