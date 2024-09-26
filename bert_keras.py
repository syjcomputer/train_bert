#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
from bert4keras.backend import keras
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
import os
import tensorflow as tf
from keras import regularizers
import random
import logging

from keras.layers import Dropout, Dense
from configs.keras_config import *
from configs.logger_config import setup_logger
from utils import load_data


class data_generator(DataGenerator):
    """数据生成器
    """
    def __init__(self, data, tokenizer, maxlen, batch_size=32, buffer_size=None):
        DataGenerator.__init__(self, data, batch_size, buffer_size)

        ###################        #self.data = data
        #self.batch_size = batch_size
        #self.buffer_size = buffer_size
        self.tokenizer = tokenizer
        self.maxlen = maxlen

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, num) in self.sample(random):
            token_ids, segment_ids = self.tokenizer.encode(text, maxlen=maxlen)
            labels = [0] * len(token_ids)

            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([num])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []

def evaluate(data, model):
    total, right = 0., 0.
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]

        for i in range(len(y_true)):
            if y_pred[i] == y_true[i]:
                right += 1
        total += len(y_true)
    return right / total


class Evaluator(keras.callbacks.Callback):
    def __init__(self, vaild_generator, model):
        self.best_val_acc = 0.
        self.vaild_generator = vaild_generator
        self.model = model

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(self.vaild_generator, self.model)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            # 模型存储
            self.model.save_weights(cur_path + 'best_model.weights')
            # self.model.save_weights(cur_path + 'best_model.weights')
            #self.model.save(cur_path + 'best_model1.h5')
            # self.model.save(cur_path + 'best_model.h5')
        logger.info(f"Epoch {epoch}, Val_acc: {val_acc}")
        logger.info(f"Epoch {epoch}, Best_val_acc:{self.best_val_acc}")

def finetune_train():
    # 1.加载数据
    train_data = load_data(cur_path + 'val.json') # [(text, label), (...),...]
    vaild_data = load_data(cur_path + 'test.json')
    test_data = load_data(cur_path + 'test.json')
    logger.info(f"Train nums: {len(train_data)}; Test nums: {len(test_data)}")

    # 2.建立分词器
    tokenizer = Tokenizer(dict_path, do_lower_case=True) # dict_path is different for chinese and English

    train_generator = data_generator(train_data, tokenizer, maxlen, batch_size)
    test_generator = data_generator(test_data, tokenizer, maxlen, batch_size)

    # 3.build models
    if model_name=="bert":
        bert = build_transformer_model(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            with_pool=True,
            return_keras_model=False,
        )

        logger.info(bert.initializer)
        classify_output = Dropout(rate=0, name='final_Dropout')(bert.model.output)
    # elif model_name=="roberta":
    #     roberta_model = build_bert_model(config_path,
    #                                      checkpoint_path,
    #                                      roberta=True)  # 建立模型，加载权重
    #     classify_output = Lambda(lambda x: x[:, 0])(roberta_model.output)  # 取出[CLS]对应的向量用来做分类

    classify_output = Dense(units=classify_num_labels, # units是输出层维度
                            activation='softmax',
                            name='classify_output',
                            kernel_initializer=bert.initializer,
                            kernel_regularizer=regularizers.l2(0.01)
                            )(classify_output)
    model = keras.models.Model(bert.model.input, classify_output)

    # 4. 加载
    model_path = cur_path + 'best_model.weights'
    model.load_weights(model_path)

    # 5.冻结除最后一层以外的参数, keras直接关闭梯度回传就行了
    for layer in model.layers[:-1]:
        layer.trainable = False

    logger.info(model.summary())
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(learning_rate),
        metrics=['accuracy'],
    )

    # 6. 训练
    evaluator = Evaluator(test_generator, model) # 测试
    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        # class_weight = 'auto',
        callbacks=[evaluator]
    )

def train():
    # 1.加载数据
    train_data = load_data(cur_path + 'train.json')  # [(text, label), (...),...]
    vaild_data = load_data(cur_path + 'val.json')
    test_data = load_data(cur_path + 'test.json')
    logger.info(f"Train nums: {len(train_data)}; Val nums:{len(vaild_data)}; Test nums: {len(test_data)}")

    # 2.建立分词器
    tokenizer = Tokenizer(dict_path, do_lower_case=True)  # dict_path is different for chinese and English

    train_generator = data_generator(train_data, tokenizer, maxlen, batch_size)
    test_generator = data_generator(test_data, tokenizer, maxlen, batch_size)

    # 3.build models
    if model_name == "bert":
        bert = build_transformer_model(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            with_pool=True,
            return_keras_model=False,
        )

        logger.info(bert.initializer)
        classify_output = Dropout(rate=0, name='final_Dropout')(bert.model.output)
    # elif model_name=="roberta":
    #     roberta_model = build_bert_model(config_path,
    #                                      checkpoint_path,
    #                                      roberta=True)  # 建立模型，加载权重
    #     classify_output = Lambda(lambda x: x[:, 0])(roberta_model.output)  # 取出[CLS]对应的向量用来做分类

    classify_output = Dense(units=classify_num_labels,  # units是输出层维度
                            activation='softmax',
                            name='classify_output',
                            kernel_initializer=bert.initializer,
                            kernel_regularizer=regularizers.l2(0.01)
                            )(classify_output)
    model = keras.models.Model(bert.model.input, classify_output)

    logger.info(model.summary())
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(learning_rate),
        metrics=['accuracy'],
    )

    # 4. 训练
    evaluator = Evaluator(test_generator, model)  # 测试
    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        # class_weight = 'auto',
        callbacks=[evaluator]
    )

if __name__ == '__main__':

    # 配置
    config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=num_cores,
                                      inter_op_parallelism_threads=num_cores,
                                      gpu_options=tf.compat.v1.GPUOptions(
                                          visible_device_list="0",  # choose GPU device number
                                          allow_growth=True
                                      ),
                                      allow_soft_placement=True,
                                      device_count={'CPU': 2})
    session = tf.compat.v1.Session(config=config)
    # K.set_session(session)
    tf.compat.v1.keras.backend.set_session(session)

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定 GPU 设备
    os.environ["TF_KERAS"] = "1"  # 使用 TensorFlow Keras 后端

    # 日志
    setup_logger(log_file_path, log_file_path2)
    logger = logging.getLogger("loggers")

    if model_name=="bert":
        finetune_train() # 冻结最后一层微调
        # train() # 训练
        # get_test_acc(risk=False)
