#! /user/bin/evn python
# -*- coding:utf8 -*-
import os
import datetime


# 加载数据并解析为 {pid: [cat1, cat2, ...]}
def load_score_data(file_path, target_pids):
    data_map = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        next(f)  # 跳过表头
        for line in f:
            if not line.strip(): continue
            pid, scores = line.strip().split(';')
            if pid in target_pids:
                categories = [item.split(':')[0] for item in scores.split(',') if ':' in item]
                data_map[pid] = categories
    return data_map


def calculate_pair_similarity(base_dict, target_dict):
    total_matches = 0
    total_paragraphs = 0
    common_pids = set(base_dict.keys()) & set(target_dict.keys())
    for pid in common_pids:
        base_list = base_dict[pid]
        target_list = target_dict[pid]
        min_len = min(len(base_list), len(target_list))
        for i in range(min_len):
            if base_list[i] == target_list[i]:
                total_matches += 1
        total_paragraphs += min_len
    return total_matches / total_paragraphs if total_paragraphs > 0 else 0


def run_full_comparison(pid_list_file, gpt3_file, gpt5_file, gemini_file):
    # 1. 获取目标 PID 列表
    with open(pid_list_file, 'r', encoding='utf-8') as f:
        target_pids = {line.strip().split(',')[0] for line in f if line.strip()}
    # 2. 加载所有模型数据
    print("正在加载数据...")
    gpt3_data = load_score_data(gpt3_file, target_pids)
    gpt5_data = load_score_data(gpt5_file, target_pids)
    gemini_data = load_score_data(gemini_file, target_pids)
    # 3. 计算相似度
    sim_gpt5_vs_gpt3 = calculate_pair_similarity(gpt3_data, gpt5_data)
    sim_gemini_vs_gpt3 = calculate_pair_similarity(gpt3_data, gemini_data)
    # 4. 打印结果
    print(f"基于段落层级的对比结果 (PID总数: {len(target_pids)})")
    print(f"GPT-5 vs GPT-3 相似度: {sim_gpt5_vs_gpt3:.2%}")
    print(f"Gemini vs GPT-3 相似度: {sim_gemini_vs_gpt3:.2%}")


def subdataset(pid_list_file, model_file, output_file):
    with open(pid_list_file, 'r', encoding='utf-8') as f:
        target_pids = {line.strip().split(',')[0] for line in f if line.strip()}
    with open(model_file, 'r', encoding='utf-8') as fin, open(output_file, 'w') as fout:
        fout.write('s2orc_paper_id;writing_score\n')
        for line in fin:
            pid, writing_score = line.strip().split(';')
            if pid in target_pids:
                fout.write(line)


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    gpt_path = "/Volumes/SanDisk_2TB/s2orc_field/"
    FIELD = "physics"
    MODEL = "gemini-3-flash-preview-nothinking"
    gpt52_writing_score = f"test_set/{FIELD}_writing_score_gpt-5.2.txt"
    gpt35_writing_score = f"writing_score/Physics_writing_score.txt"
    gemini3_writing_score = f"test_set/{FIELD}_writing_score_gemini-3-flash-preview-nothinking.txt"
    subdataset(
        f"{FIELD}_compared_score.txt",
        os.path.join(gpt_path, gemini3_writing_score),
        os.path.join(gpt_path, f"test_set/{FIELD}_writing_score_gemini-3-flash-preview-nothinking_subsection.txt")
    )
    run_full_comparison(
        f"{FIELD}_compared_score.txt",
        os.path.join(gpt_path, "test_set/writing_score_gpt_3.5.txt"),
        os.path.join(gpt_path, f"test_set/writing_score_gpt-5.2.txt"),
        os.path.join(gpt_path, f"test_set/writing_score_gemini-3.txt"),
    )
    end_time = datetime.datetime.now()
    print('It takes {} seconds.'.format((end_time - start_time).seconds))
    print('Done LLM_predict_similarity!')
