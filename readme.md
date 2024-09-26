本项目提供了bert系列模型的使用代码，包含keras和torch两种框架代码，下游任务为多分类任务。

# 1. Data Prepare

```
data-|-train.json
     |-test.json
     |-val.json
```
数据文件为json格式，包含"text"和"label"。

# 2. keras框架
目前keras框架实现了训练和微调两种模式，但只写了bert的代码，使用keras框架需要把tf版本的bert模型下载到本地。
英语文本的话需要下载uncased_L-12_H-768_A-12文件夹，在配置文件中配置以下内容：
```
    config_path = 'uncased_L-12_H-768_A-12/bert_config.json'
    checkpoint_path = 'uncased_L-12_H-768_A-12/bert_model.ckpt'
    dict_path = 'uncased_L-12_H-768_A-12/vocab.txt'
```
bert_keras.py内包含训练和微调的代码：
> 1. 训练bert调用**train()**
> 2. 微调bert调用**finetune_train()**，微调bert使用的是val.json进行微调，冻结除最后一层线性层以外的参数。

```python bert_keras.py```需要配置参数，在**configs/keras_config.py**配置，必须配置的参数如下：

* **model_name**： 默认是bert，目前只实现了bert模型代码，可自行添加roberta等代码
* **classify_num_labels**：多分类的类别数
* **cur_path**：模型以及文件路径
* **log_file_path**：日志文件路径

**注意**：train和finetune都用的一个evaluator，训练得到的模型文件默认存储到{cur_path}/best_model.weights，如果不希望微调后覆盖原模型文件，需要注释掉模型存储或者修改存储位置

# 3. pytorch
实现了训练与微调两部分代码，包含bert，roberta，xlnet三种模型，具体的模型文件可以在[huggingface官网](https://huggingface.co/models)下载,然后在配置文件中配置路径
> 1. 训练bert调用**train_bert()**
> 2. 微调bert调用**finetune()**，微调bert使用的是val.json进行微调，冻结除最后一层线性层以外的参数。

```python bert_torch.py```需要配置参数，在**configs/torch_config.py**配置，必须配置的参数如下：
* **model_name**： 默认是bert
* **classify_num_labels**：多分类的类别数
* **cur_path**：模型以及文件路径
* **log_file_path2**：日志文件路径

**注意**：train和finetune都用的一个evaluator，训练得到的模型文件默认存储到{cur_path}/best_model.h5，如果不希望微调后覆盖原模型文件，需要注释掉模型存储或者修改存储位置