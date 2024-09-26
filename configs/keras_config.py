import torch

# 基本信息
seed = 10
num_cores = 10
maxlen = 150
epochs = 100
batch_size = 256
learning_rate = 0.01   # 4e-5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_name = 'bert'
classify_num_labels = 3
cur_path = '/home/8t/syj/UST-data/syj_computer/'
log_file_path = "keras.log"
log_file_path2 = "torch.log"

if model_name=="bert":
    config_path = '/home/ssd1/syj/risk_code/zh_get_risk_dataset/uncased_L-12_H-768_A-12/bert_config.json'
    checkpoint_path = '/home/ssd1/syj/risk_code/zh_get_risk_dataset/uncased_L-12_H-768_A-12/bert_model.ckpt'
    dict_path = '/home/ssd1/syj/risk_code/zh_get_risk_dataset/uncased_L-12_H-768_A-12/vocab.txt'
elif model_name=="roberta":
    dict_path = "/home/ssd1/syj/risk_code/zh_get_risk_dataset/xlnet"
