#! /usr/bin/env python
# -*- coding:utf8 -*-
import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, f1_score, cohen_kappa_score


def svm_tfidf():
    df = pd.read_csv("evaluation_gold.csv")
    print("数据读取成功，前 5 行如下：")
    print(df.head())
    class_counts = df['label'].value_counts()
    print('class_counts', class_counts)
    # 2. 划分训练集和测试集 我们不需要 paper_id 参与训练，所以只取 text 和 label 366
    X = df['paragraph_text'].astype(str)
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y  # stratify 保证类别分布一致
    )
    # 3. 特征提取：TF-IDF
    tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words='english', max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    # # 4. 训练 SVM 模型
    svm_model = SVC(kernel='linear', C=1.0, probability=True)
    svm_model.fit(X_train_tfidf, y_train)
    # # 5. 预测与评估
    y_pred = svm_model.predict(X_test_tfidf)
    print("\n" + "=" * 30)
    print("--- SVM 分类报告 ---")
    print(classification_report(y_test, y_pred))
    print(f"总体准确率 (Accuracy): {accuracy_score(y_test, y_pred):.2%}")
    print("=" * 30)
    # 假设 y_test 是真实标签，y_pred 是 SVM 预测的标签
    kappa_svm = cohen_kappa_score(y_test, y_pred)
    print(f"--- SVM 性能评估 ---")
    print(classification_report(y_test, y_pred))
    print(f"SVM Cohen's Kappa: {kappa_svm:.4f}")


def compare_gpt3_human():
    # --- 1. 处理 GPT 的 TXT 文件 ---
    gpt_rows = []
    with open('writing_score_evaluation_gpt3.txt', 'r', encoding='utf-8') as f:
        next(f)  # 跳过表头
        for line in f:
            pid, scores = line.strip().split(';')
            paragraph_list = scores.split(',')
            for i, p_info in enumerate(paragraph_list):
                if ':' in p_info:
                    func = p_info.split(':')[0]  # 提取 function
                    gpt_rows.append({
                        'paper_id': str(pid),
                        'para_idx': i,  # 段落序号（从0开始）
                        'gpt_label': str(func)
                    })
    df_gpt = pd.DataFrame(gpt_rows)
    print('df_gpt', df_gpt)
    # --- 2. 处理人类标注的 CSV 文件 ---
    df_human = pd.read_csv('evaluation_gold.csv', sep=',', quotechar='"')
    df_human['paper_id'] = df_human['paper_id'].astype(str)
    df_human['para_idx'] = df_human.groupby('paper_id').cumcount()
    # --- 3. 合并数据 (Inner Join) --- 通过 paper_id 和段落序号进行精确匹配
    df_merge = pd.merge(
        df_human, df_gpt,
        on=['paper_id', 'para_idx'],
        how='inner'
    )
    # --- 4. 评价函数 ---
    y_true = df_merge['label'].astype(str)
    y_pred = df_merge['gpt_label'].astype(str)
    print("=== GPT vs Human 详细分类报告 ===")
    print(classification_report(y_true, y_pred))
    accuracy = (y_true == y_pred).mean()
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    kappa = cohen_kappa_score(y_true, y_pred)
    print(f"准确率 (Accuracy): {accuracy:.2%}")
    print(f"Macro F1 Score: {macro_f1:.4f}")
    print(f"Cohen's Kappa: {kappa:.4f}")


# human_df: 包含 paper_id, label (人类给的类别标签)
# gpt_df: 包含 s2orc_paper_id, writing_score (格式为 1:3,2:2...)
def analyze_label_discrepancy():
    human_df = pd.read_csv("evaluation_gold.csv")
    gpt_df = pd.read_csv("writing_score_evaluation_gpt3.txt", sep=';')
    # 1. 解析 GPT 标签字符串
    gpt_records = []
    for _, row in gpt_df.iterrows():
        paper_id = str(row['s2orc_paper_id'])
        scores = str(row['writing_score']).split(',')
        for idx, item in enumerate(scores):
            if ':' in item:
                label_type = item.split(':')[0]
                gpt_records.append({'paper_id': paper_id, 'paragraph_index': idx, 'gpt_label': int(label_type)})
    gpt_parsed = pd.DataFrame(gpt_records)
    # 2. 为人类标签也加上段落索引 (假设 human_df 的顺序与 GPT 字符串顺序一致)
    human_df['paper_id'] = human_df['paper_id'].astype(str)
    human_df['paragraph_index'] = human_df.groupby('paper_id').cumcount()
    # 3. 合并数据
    merged = pd.merge(human_df, gpt_parsed, on=['paper_id', 'paragraph_index'])
    # 4. 生成混淆矩阵 (Discrepancy Matrix) 行代表人类标签，列代表 GPT 标签
    discrepancy_matrix = pd.crosstab(merged['label'], merged['gpt_label'], rownames=['Human'], colnames=['GPT-3'])
    # 5. 计算百分比分布 (每一行人类标签被 GPT 分类成了什么的比例)
    discrepancy_pct = discrepancy_matrix.div(discrepancy_matrix.sum(axis=1), axis=0) * 100
    print('discrepancy_matrix', discrepancy_matrix)
    print('discrepancy_pct', discrepancy_pct)


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    start_time = datetime.datetime.now()
    svm_tfidf()
    compare_gpt3_human()
    analyze_label_discrepancy()
    end_time = datetime.datetime.now()
    print('It takes {} seconds.'.format((end_time - start_time).seconds))
    print('Done SVM!')
