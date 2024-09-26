from transformers import RobertaModel, RobertaTokenizer, BertModel, \
    BertTokenizer, AutoModelForSequenceClassification, \
    XLNetTokenizer,XLNetForSequenceClassification, XLNetModel
import torch
import random
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import os
import numpy as np
from sklearn import metrics
from torch.utils.data import Dataset

from configs.torch_config import *
from utils import *
from configs.logger_config import *



class textDataset(Dataset):
    def __init__(self, datas, labels):
        super().__init__()
        self.texts = datas
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        return text, label

def train(train_loader, optimizer, model, criterion, tokenizer):
    # 训练
    model.train()

    total_loss = 0
    acc = 0
    batch_num = len(train_loader)
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        input_ids, attention_mask, y = data

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        y = y.to(device)

        if model_name == 'bert' or model_name == 'roberta':
            out = model(input_ids, token_type_ids = None,attention_mask = attention_mask, labels = y)
            loss, logits = out[0], out[1]
        elif model_name=='xlnet':
            out = model(input_ids, token_type_ids=None, attention_mask=attention_mask)
            logits = out.logits    # (batch, label_nums)
            loss = criterion(logits, y.squeeze(1))

        #loss = criterion(out, y.long())
        total_loss += loss

        loss.backward()
        optimizer.step()

        predictions = np.argmax(logits.cpu().detach().numpy(), axis=-1)
        acc += metrics.accuracy_score(y.cpu(), predictions)
    acc = acc / batch_num

    return total_loss, acc


def test(test_loader, criterion, model, tokenizer):
    # 测试
    model.eval()

    with torch.no_grad():
        total_loss = 0
        acc = 0
        batch_num = len(test_loader)
        for i, data in enumerate(test_loader):
            input_ids,attention_mask, y = data

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            y = y.to(device)

            if model_name == 'bert' or model_name == 'roberta':
                out = model(input_ids, token_type_ids=None, attention_mask=attention_mask, labels=y)
                loss, logits = out[0], out[1]
            elif model_name == 'xlnet':
                out = model(input_ids, token_type_ids=None, attention_mask=attention_mask)
                logits = out.logits  # (batch, label_nums)
                loss = criterion(logits, y.squeeze(1))

            total_loss += loss

            predictions = np.argmax(logits.cpu().detach().numpy(), axis=-1)
            acc += metrics.accuracy_score(y.cpu(), predictions)
        acc = acc / batch_num

    return total_loss, acc


def get_dataloader(tokenizer, text, label):
    input_labels = torch.unsqueeze(torch.tensor(label),dim=1)
    res = tokenizer(text, padding=True,truncation=True,max_length=maxlen)
    data_set=TensorDataset(torch.LongTensor(res['input_ids']),
                           torch.LongTensor(res['attention_mask']),
                           torch.LongTensor(input_labels))
    data_loader = DataLoader(dataset=data_set,
                             batch_size=batch_size,
                             shuffle=True)
    return data_loader


def train_bert():
    # 1.load data
    train_text, train_label, _ = load_data1(cur_path + 'train.json')  # [(text, label), (...),...]

    valid_text, valid_label, _ = load_data1(cur_path + 'val.json')

    test_text, test_label, _ = load_data1(cur_path + 'test.json')

    logger.info(f"Train size:{len(train_text)}; Val size:{len(valid_text)}; Test size:{len(test_text)}")

    # 2. 模型选择
    if model_name == 'bert':
        tokenizer = BertTokenizer.from_pretrained(f'{dict_path}/')
        model = AutoModelForSequenceClassification.from_pretrained(f'{dict_path}/', num_labels = classify_num_labels)
    elif model_name == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained(f"{dict_path}")
        model = AutoModelForSequenceClassification.from_pretrained(f'{dict_path}/', num_labels=classify_num_labels)
    elif model_name == 'xlnet':
        tokenizer = XLNetTokenizer.from_pretrained(f"{dict_path}")
        model = XLNetForSequenceClassification.from_pretrained(f"{dict_path}/", num_labels=classify_num_labels)
    else:
        logger.info("不存在这个模型")
        raise ExistError("不存在这个模型")

    train_dataloader = get_dataloader(tokenizer, train_text, train_label)
    test_dataloader = get_dataloader(tokenizer, test_text, test_label)

    model = model.to(device)

    model.apply(weight_init)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.01)

    val_accs = 0
    best_acc = 0

    for epoch in tqdm(range(epochs), desc='epochs'):
        train_loss, train_acc = train(train_dataloader, optimizer, model, criterion, tokenizer)
        test_loss, test_acc = test(test_dataloader, criterion, model, tokenizer)

        val_accs += test_acc
        logger.info(f'Epoch {epoch}, Train Loss:{train_loss}, Acc:{train_acc}; Val Loss:{test_loss}, Acc:{test_acc}')

        # save model parameters
        # save_path = f"{path}/checkpoint.pt"
        if best_acc < test_acc:
            best_acc = test_acc
            model.save_pretrained(f"{cur_path}/best_model.h5")
            # torch.save(model.state_dict(), save_path)
            print("New model has been saved!")

def fine_tune():
    # 1. load data
    train_text, train_label, _ = load_data1(cur_path + 'train.json')  # [(text, label), (...),...]

    valid_text, valid_label, _ = load_data1(cur_path + 'val.json')

    test_text, test_label, _ = load_data1(cur_path + 'test.json')

    logger.info(f"Train size:{len(train_text)}; Val size:{len(valid_text)}; Test size:{len(test_text)}")

    # 2. 模型选择
    if model_name == 'bert':
        tokenizer = BertTokenizer.from_pretrained(f'{dict_path}/')
        model = AutoModelForSequenceClassification.from_pretrained(f'{dict_path}/', num_labels=classify_num_labels)
    elif model_name == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained(f"{dict_path}")
        model = AutoModelForSequenceClassification.from_pretrained(f'{dict_path}/', num_labels=classify_num_labels)
    elif model_name == 'xlnet':
        tokenizer = XLNetTokenizer.from_pretrained(f"{dict_path}")
        model = XLNetForSequenceClassification.from_pretrained(f"{dict_path}/", num_labels=classify_num_labels)
    else:
        logger.info("不存在这个模型")
        raise ExistError("不存在这个模型")

    # 3. 冻结除最后一层以外的
    i=1
    for param in model.parameters():
        if i==len(list(model.parameters())):
            break
        i+=1
        param.requires_grad=False


    train_dataloader = get_dataloader(tokenizer, valid_text, valid_label)
    test_dataloader = get_dataloader(tokenizer, test_text, test_label)

    model = model.to(device)

    # model.apply(weight_init)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.01)

    val_accs = 0
    best_acc = 0

    for epoch in tqdm(range(epochs), desc='epochs'):
        train_loss, train_acc = train(train_dataloader, optimizer, model, criterion, tokenizer)
        test_loss, test_acc = test(test_dataloader, criterion, model, tokenizer)

        val_accs += test_acc
        logger.info(f'Epoch {epoch}, Train Loss:{train_loss}, Acc:{train_acc}; Val Loss:{test_loss}, Acc:{test_acc}')

        # save model parameters
        # save_path = f"{path}/checkpoint.pt"
        if best_acc < test_acc:
            best_acc = test_acc
            model.save_pretrained(f"{cur_path}/best_model.h5")
            # torch.save(model.state_dict(), save_path)
            print("New model has been saved!")

if __name__ == "__main__":

    # 配置
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 下面老是报错 shape 不一致

    # 日志
    setup_logger(log_file_path, log_file_path2)
    logger = logging.getLogger("loggers2")

    # 训练 or 微调
    # train_bert()
    fine_tune()