#! /user/bin/evn python
# -*- coding:utf8 -*-
import pandas as pd
import datetime
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, cohen_kappa_score
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import numpy as np


# 1. 准备数据集格式
class DiscourseDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# 定义评估函数
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    macro_f1 = f1_score(labels, predictions, average='macro')
    weighted_f1 = f1_score(labels, predictions, average='weighted')
    return {'accuracy': acc, 'macro_f1': macro_f1, 'weighted_f1': weighted_f1}


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    # 2. 加载并清洗数据 (沿用你之前的逻辑)
    df = pd.read_csv('evaluation_gold.csv', sep=',', quotechar='"')
    df['label'] = df['label'].astype(str).str.strip('"').astype(int)
    df['paragraph_text'] = df['paragraph_text'].astype(str).str.strip('"')
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df['paragraph_text'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42, stratify=df['label']
    )
    # 3. 加载 BERT 分词器和模型
    model_name = 'bert-base-uncased'  # 学术论文通用基础模型
    tokenizer = BertTokenizer.from_pretrained(model_name)
    num_labels = len(df['label'].unique())
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # 4. 编码数据
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=512)
    train_dataset = DiscourseDataset(train_encodings, train_labels)
    test_dataset = DiscourseDataset(test_encodings, test_labels)
    # 5. 设置训练参数
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=5,  # 迭代次数
        per_device_train_batch_size=4,  # 显存不够就调小
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        eval_strategy="epoch",  # 注意：这里从 evaluation_strategy 改为 eval_strategy
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )
    trainer.train()
    # 获取预测结果
    preds = trainer.predict(test_dataset)
    pred_labels = np.argmax(preds.predictions, axis=-1)
    print("--- BERT 最终分类报告 ---")
    print(classification_report(test_labels, pred_labels))
    # 基于现有的 test_labels 和 pred_labels 计算 Kappa
    kappa = cohen_kappa_score(test_labels, pred_labels)
    print(f"--- BERT 稳定性指标 ---")
    print(f"Cohen's Kappa: {kappa:.4f}")
    end_time = datetime.datetime.now()
    print('It takes {} seconds.'.format((end_time - start_time).seconds))
    print('Done DL_evaluation!')
