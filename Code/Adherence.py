import pandas as pd
import datetime
from scipy.stats import spearmanr


class Adherence(object):
    def __init__(self):
        self.common_path = "Code&data/"
        self.writing_score_data_path = self.common_path + "ChatGPT_writing_score/"
        self.paper_data_path = self.common_path + "MAG_paper_author_topic/"
        self.journal_data_path = self.common_path + "MAG_journal_info/"
        self.output = self.common_path + "Analysis_output/"

    # coverage
    def require_coverage(self, field):
        df = pd.read_csv(self.writing_score_data_path + field + "_writing_score_normalization_full.txt", sep=';')
        steps = ["IT", "DB", "ERQ", "SO", "SF", "MO"]
        df["coverage"] = round(df[steps].gt(0).sum(axis=1) / len(steps), 2)
        df["total_effort"] = df[steps].sum(axis=1)
        threshold = df["total_effort"] * 0.05  # 每篇文章的 5% effort
        weighted_cov = []
        for i, row in df.iterrows():
            count = 0
            for s in steps:
                if row[s] >= threshold[i]:
                    count += 1
            weighted_cov.append(round(count / len(steps), 2))
        df["coverage_weighted"] = weighted_cov
        df[["s2orc_paper_id", "coverage", "coverage_weighted", ]] \
            .to_csv(self.output + field + "_coverage.txt", index=False)
        print(df[["s2orc_paper_id", "coverage", "coverage_weighted"]].head())

    # 根据片段得分生成实际序列（合并连续重复步骤）
    def get_actual_seq(self, row, steps, segments):
        seq = []
        for seg in segments:
            # 找到该片段得分最高的步骤
            scores = {step: row[f"{step}_{seg}"] for step in steps}
            best_step = max(scores, key=scores.get)
            if scores[best_step] > 0:  # 只考虑得分>0的步骤
                seq.append(best_step)
        # 合并相邻相同的步骤
        merged_seq = []
        for s in seq:
            if not merged_seq or merged_seq[-1] != s:
                merged_seq.append(s)
        return merged_seq

    # 根据每个步骤在6个片段中的得分计算实际六步序列，并计算与理想序列的距离
    def calculate_six_step_distance(self, field):
        pd.set_option('display.max_columns', None)
        df = pd.read_csv(self.writing_score_data_path + field + "_writing_score_normalization_full.txt", sep=';')
        steps = ["IT", "DB", "ERQ", "SO", "SF", "MO"]
        segments = ["0_17", "17_34", "34_50", "50_67", "67_84", "84_100"]
        ideal_seq = steps
        actual_seqs = []
        spearman_values = []
        for idx, row in df.iterrows():
            actual_seq = []
            for seg in segments:
                scores = {step: row[f"{step}_{seg}"] for step in steps}
                best_step = max(scores, key=scores.get)
                actual_seq.append(best_step)
            actual_seqs.append(actual_seq)
            # Spearman 排序相关系数
            ideal_ranks = {step: i + 1 for i, step in enumerate(ideal_seq)}
            actual_ranks = [ideal_ranks[step] for step in actual_seq]
            rho, _ = spearmanr(list(range(1, len(ideal_seq) + 1)), actual_ranks)
            spearman_values.append(rho)
        df["actual_seq"] = actual_seqs
        df["spearman_seq"] = spearman_values
        df[['s2orc_paper_id', 'actual_seq', 'spearman_seq']]\
            .to_csv(self.output + field + "_sequence_distance.txt", sep=";", index=False)
        print(df[['s2orc_paper_id', 'actual_seq', 'spearman_seq']].head())

    # 统计写作不清楚的段落的比例
    def unclear_ratio(self, field):
        df = pd.read_csv(self.writing_score_data_path + field + '_writing_score.txt', sep=';')
        results = []
        for _, row in df.iterrows():
            paper_id = row['s2orc_paper_id']
            scores = str(row['writing_scores']).split(',')
            total = 0
            unclear = 0
            for item in scores:
                if ':' not in item:
                    continue
                step, count = item.split(':')
                count = int(count)
                total += count
                if step == "0":
                    unclear += count
            ratio = unclear / total if total > 0 else 0
            results.append({"s2orc_paper_id": paper_id,
                            "unclear_count": unclear,
                            "total_count": total,
                            "unclear_ratio": ratio})
        result_df = pd.DataFrame(results)
        result_df.to_csv(self.output + field + '_unclear_ratio.txt', sep=';', index=False)
        print(f"field {field} finish!")

    def connect_info(self, field):
        paper_dict = {}
        with open(self.paper_data_path + field + '_paper_author_topic.txt', 'r') as fin:
            for line in fin:
                line = line.strip().split(';')
                s2orc_paper_id, mag_id, year, author_num, citation_count, reference_count, topic_size, \
                first_author_prod, first_author_h_index, last_author_index, last_author_prod, last_author_h_index, \
                title_len = line
                if mag_id != 'mag_id':
                    paper_dict[s2orc_paper_id] = {
                        'year': year, 'author_num': author_num, 'citation_count': citation_count,
                        'topic_size': topic_size, 'first_author_prod': first_author_prod,
                        'first_author_h_index': first_author_h_index, 'last_author_prod': last_author_prod,
                        'last_author_h_index': last_author_h_index, 'title_len': title_len,
                    }
        with open(self.journal_data_path + field + '_journal_cc.txt', 'r') as fjour:
            for line in fjour:
                line = line.strip().split(';')
                s2orc_paper_id, mag_id, journal_id, journal_cc = line
                if s2orc_paper_id != 's2orc_paper_id' and s2orc_paper_id in paper_dict.keys():
                    paper_dict[s2orc_paper_id]['journal_cc'] = journal_cc
        with open(self.output + field + '_coverage.txt', 'r') as fcover:
            for line in fcover:
                line = line.strip().split(',')
                s2orc_paper_id, coverage, coverage_weighted = line
                if s2orc_paper_id != 's2orc_paper_id' and s2orc_paper_id in paper_dict.keys():
                    paper_dict[s2orc_paper_id]['coverage'] = coverage
                    paper_dict[s2orc_paper_id]['coverage_weighted'] = coverage_weighted
        with open(self.output + field + '_sequence_distance.txt', 'r') as fseq:
            for line in fseq:
                line = line.strip().split(';')
                s2orc_paper_id, actual_seq, spearman_seq = line
                if s2orc_paper_id != 's2orc_paper_id' and s2orc_paper_id in paper_dict.keys():
                    paper_dict[s2orc_paper_id]['spearman_seq'] = spearman_seq
        with open(self.output + field + '_unclear_ratio.txt', 'r') as funclear:
            for line in funclear:
                line = line.strip().split(';')
                s2orc_paper_id, unclear_count, total_count, unclear_ratio = line
                if s2orc_paper_id != 's2orc_paper_id' and s2orc_paper_id in paper_dict.keys():
                    paper_dict[s2orc_paper_id]['unclear_ratio'] = unclear_ratio
        with open(self.writing_score_data_path + field + '_writing_score_normalization.txt', 'r') as fws, \
                open(self.output + field + '_regression.txt', 'w') as fout:
            fout.write('s2orc_paper_id;'
                       'year;author_num;topic_size;first_author_prod;first_author_h_index;'
                       'last_author_prod;last_author_h_index;title_len;citation_count;journal_cc;'
                       'coverage;coverage_weighted;spearman_seq;unclear_ratio;'
                       'len_paragraph;IT;DB;ERQ;SO;SF;MO;'
                       'IT_0_17;IT_17_34;IT_34_50;IT_50_67;IT_67_84;IT_84_100;'
                       'DB_0_17;DB_17_34;DB_34_50;DB_50_67;DB_67_84;DB_84_100;ERQ_0_17;ERQ_17_34;ERQ_34_50;ERQ_50_67;'
                       'ERQ_67_84;ERQ_84_100;SO_0_17;SO_17_34;SO_34_50;SO_50_67;SO_67_84;SO_84_100;SF_0_17;SF_17_34;'
                       'SF_34_50;SF_50_67;SF_67_84;SF_84_100;MO_0_17;MO_17_34;MO_34_50;MO_50_67;MO_67_84;MO_84_100\n')
            for line in fws:
                line = line.strip().split(';')
                p_id = line[0]
                if p_id != 's2orc_paper_id' and p_id in paper_dict.keys():
                    other_fields = ';'.join(line[1:])
                    fout.write('{0};{1};{2};{3};{4};{5};{6};{7};{8};{9};{10};{11};{12};{13};{14};{15}\n'.format(
                        p_id, paper_dict[p_id]['year'], paper_dict[p_id]['author_num'], paper_dict[p_id]['topic_size'],
                        paper_dict[p_id]['first_author_prod'], paper_dict[p_id]['first_author_h_index'],
                        paper_dict[p_id]['last_author_prod'], paper_dict[p_id]['last_author_h_index'],
                        paper_dict[p_id]['title_len'], paper_dict[p_id]['citation_count'],
                        paper_dict[p_id]['journal_cc'], paper_dict[p_id]['coverage'],
                        paper_dict[p_id]['coverage_weighted'], paper_dict[p_id]['spearman_seq'],
                        paper_dict[p_id]['unclear_ratio'], other_fields
                    ))


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    ad = Adherence()
    for field in ['Biology', 'Computer Science', 'Economics', 'Medicine', 'Physics', 'Mathematics']:
        ad.require_coverage(field)
        ad.calculate_six_step_distance(field)
        ad.connect_info(field)
    end_time = datetime.datetime.now()
    print('It takes {} seconds.'.format((end_time - start_time).seconds))
    print('Done Adherence!')
