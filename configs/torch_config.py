import torch

# 基本信息
seed=10
num_cores = 10
maxlen = 150
epochs = 30
batch_size = 64
learning_rate = 4e-3  # 4e-5

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

path = f"/home/8t/syj/train_models/bert/roberta"
cur_path = ""
log_file_path = "keras.log"
log_file_path2 = "torch.log"
model_name = 'bert'
classify_num_labels = 4


if model_name=="bert":
    config_path = 'uncased_L-12_H-768_A-12/bert_config.json'
    checkpoint_path = 'uncased_L-12_H-768_A-12/bert_model.ckpt'
    dict_path = 'uncased_L-12_H-768_A-12/vocab.txt'
elif model_name=="roberta":
    dict_path = "/home/ssd1/syj/risk_code/zh_get_risk_dataset/roberta"